# ---------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# HANA RAG CODE

from json import load
import os
import re
import io
import logging
from typing import List, Dict
from pathlib import Path

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sqlalchemy import create_engine, text
import streamlit as st
import pandas as pd
from hdbcli import dbapi
from sqlalchemy.engine import URL

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.tools.base import Tool

from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()


def extract_between_tags(text: str, tag_name: str) -> List[str]:
    """Extract content between specified XML-style tags"""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, text, re.DOTALL)
    print(matches)
    return matches


class DatabaseManager:
    def __init__(self: str, address, port, user, password):
        """Initialize database connection"""

        self.conn = dbapi.connect(
            address=address,
            port=port,
            user=user,
            password=password
        )
        self.table_name = "excel_data"

    def store_dataframe(self, file_path: str, if_exists: str = 'append') -> None:
        """Store DataFrame in excel_data table using bulk insert"""
        try:
            df = pd.read_excel(file_path)

            # Handle NaN values: replace with None to store as NULL in DB
            df = df.applymap(lambda x: None if pd.isna(x) else x)

            # Dynamically generate column definitions based on DataFrame types
            column_definitions = []
            for column, dtype in df.dtypes.items():
                # Make sure column names are quoted to avoid issues with special characters
                column_name_quoted = f'"{column}"'

                if pd.api.types.is_integer_dtype(dtype):
                    column_definitions.append(f"{column_name_quoted} INTEGER")
                elif pd.api.types.is_float_dtype(dtype):
                    column_definitions.append(f"{column_name_quoted} DECIMAL")
                elif pd.api.types.is_string_dtype(dtype):
                    # Calculate max length for the column
                    max_length = df[column].astype(str).str.len().max()
                    # Use NVARCHAR with appropriate length, default to 5000 if very long
                    length = min(max_length + 100, 5000)  # add some buffer
                    column_definitions.append(f"{column_name_quoted} NVARCHAR({length})")
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    column_definitions.append(f"{column_name_quoted} DATE")
                else:
                    column_definitions.append(f"{column_name_quoted} NVARCHAR({5000})")  # Default to TEXT

            column_definitions_str = ", ".join(column_definitions)

            with self.conn.cursor() as cursor:
                # If replace mode or table doesn't exist, create new table
                if if_exists == 'replace' or not self.table_exists():
                    if self.table_exists():
                        cursor.execute(f'DROP TABLE "{self.table_name}"')
                    cursor.execute(f'CREATE TABLE "{self.table_name}" ({column_definitions_str})')
                    self.conn.commit()

                # Bulk insert data
                column_names = ", ".join([f'"{col}"' for col in df.columns])
                placeholders = ", ".join(["?" for _ in df.columns])
                insert_query = f'INSERT INTO "{self.table_name}" ({column_names}) VALUES ({placeholders})'

                # Convert DataFrame to list of tuples for bulk insert
                data = [tuple(row) for row in df.values]
                cursor.executemany(insert_query, data)
                self.conn.commit()

            logger.info(f"Data stored successfully in {self.table_name}")
        except Exception as e:
            logger.error(f"Error storing DataFrame: {str(e)}")
            raise

    def table_exists(self) -> bool:
        """Check if table exists"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM SYS.TABLES WHERE table_name = '{self.table_name}'")
                result = cursor.fetchone()
                return result[0] > 0
        except Exception as e:
            logger.error(f"Error checking table existence: {str(e)}")
            return False

    def execute_query(self, query: str) -> List[str]:
        """Execute SQL query"""
        try:
            with self.conn.cursor() as cursor:
                query = extract_between_tags(query, "sql")[0]
                print(query)
                # cleaned_query = query.strip().strip("'")
                # print("Cleaned query:", cleaned_query)  # Debug print
                cursor.execute(query)
                result_set = cursor.fetchall()
                return [str(row) for row in result_set]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return str(e)
            # raise

    def get_metadata(self):
        """Get metadata for the table columns"""
        try:
            with self.conn.cursor() as cursor:
                query = f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE
                FROM TABLE_COLUMNS
                WHERE SCHEMA_NAME = CURRENT_SCHEMA 
                AND TABLE_NAME = '{self.table_name}'
                ORDER BY POSITION
                """
                cursor.execute(query)
                metadata = cursor.fetchall()
                print("meatadata", metadata)
                return metadata
        except Exception as e:
            logger.error(f"Error retrieving metadata: {str(e)}")
            raise


class RAGSystem:
    def __init__(self, api_key: str, qdrant_url: str, qdrant_api_key: str, collection_name: str = "excel-embeddings"):
        """Initialize RAG system with OpenAI and Qdrant clients"""
        self.openai_client = OpenAI(api_key=api_key)
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name

        # Create collection if it doesn't exist
        # all_collections = [col.name for col in self.qdrant_client.get_collections().collections]
        # if self.collection_name in all_collections:
        if not self.qdrant_client.collection_exists(self.collection_name):
            # self.qdrant_client.delete_collection(collection_name=self.collection_name)
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),
            )

    def extract_columns_from_excel(
            self,
            file_path: str,
            question_col: str,
            answer_col: str,
            id_col: str,
    ) -> tuple[pd.DataFrame, List[str]]:
        """Extract and combine question-answer pairs from Excel"""
        try:
            df = pd.read_excel(file_path)
            raw_df = df.copy()

            if question_col not in df.columns or answer_col not in df.columns:
                raise ValueError(f"Required columns {question_col} and/or {answer_col} not found")

            df[question_col] = df[question_col].astype(str).str.strip()
            df[answer_col] = df[answer_col].astype(str).str.strip()
            df[id_col] = df[id_col].astype(str).str.strip()
            df.drop_duplicates([question_col], inplace=True)

            selected_columns_text = df.apply(
                lambda row: f"Ticket ID: {row[id_col]}\nQuestion: {row[question_col]}\nAnswer: {row[answer_col]}",
                axis=1
            )

            return raw_df, list(selected_columns_text)
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            raise

    def get_embeddings(
            self,
            texts: List[str],
            model: str = "text-embedding-3-small",
            batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings with batching"""
        all_embeddings = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                retries = 3
                while retries > 0:
                    try:
                        response = self.openai_client.embeddings.create(
                            input=batch,
                            model=model,
                            dimensions=100
                        )
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                        break
                    except Exception as e:
                        retries -= 1
                        if retries == 0:
                            raise e
                        logger.warning(f"Retrying embedding generation. Error: {str(e)}")
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def store_data(
            self,
            df: pd.DataFrame,
            texts: List[str],
            embeddings: List[List[float]]
    ) -> pd.DataFrame:
        """Store data in Qdrant"""
        try:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            points = [
                models.PointStruct(
                    id=id_,
                    vector=embedding,
                    payload={
                        "source": "excel",
                        "timestamp": datetime.now().isoformat(),
                        "document_id": id_,
                        "text": text
                    }
                ) for id_, text, embedding in zip(ids, texts, embeddings)
            ]

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return df
        except Exception as e:
            logger.error(f"Error storing data in Qdrant: {str(e)}")
            raise

    def query_similar(
            self,
            query_text: str,
            top_k: int = 5
    ) -> List[Dict]:
        """Query similar documents from Qdrant"""
        try:
            query_text = extract_between_tags(query_text, "rag")[0]
            print(query_text)
            query_embedding = self.get_embeddings([query_text])[0]
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )

            formatted_results = []
            for result in results:
                formatted_results.append({
                    'document': result.payload.get("text", ""),
                    'metadata': result.payload,
                    'similarity': result.score
                })
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying similar documents: {str(e)}")
            raise


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = "excel-embeddings"
    if "messages" not in st.session_state:
        st.session_state.messages = []


def process_file(
        file_path: str,
        rag_system: RAGSystem,
        db_manager: DatabaseManager,
        question_col: str,
        answer_col: str,
        id_col: str
) -> None:
    """Process a single Excel file"""

    df, texts = rag_system.extract_columns_from_excel(
        str(file_path),
        question_col,
        answer_col,
        id_col
    )

    embeddings = rag_system.get_embeddings(texts)
    df = rag_system.store_data(df, texts, embeddings)

    if db_manager.table_exists():
        db_manager.store_dataframe(file_path, if_exists='replace')
    else:
        db_manager.store_dataframe(file_path, if_exists='replace')


# For processing files
@csrf_exempt
def processing_files(request):
    logger.debug("Received a request to process files")
    if request.method == 'POST':
        try:
            uploaded_files = request.FILES.getlist('files')
            logger.debug(f"Number of files uploaded: {len(uploaded_files)}")

            api_key = os.getenv("OPENAI_API_KEY")
            # print(f"API Key: {api_key}")  # Debugging API key

            address = os.getenv("HANA_ADDRESS")
            # print(f"Address: {address}")  # Debugging HANA address
            port = os.getenv("HANA_PORT")
            # print(f"Port: {port}")  # Debugging HANA port
            user = os.getenv("HANA_USER")
            password = os.getenv("HANA_PASSWORD")

            qdrant_url = os.getenv("QDRANT_URL")
            # print(qdrant_url)
            qdrant_api = os.getenv("QDRANT_API")

            # Initialize RAGSystem and DatabaseManager
            rag_system = RAGSystem(api_key, qdrant_url, qdrant_api)
            # logger.debug("RAGSystem initialized")
            db_manager = DatabaseManager(address, port, user, password)
            # logger.debug("DatabaseManager initialized")

            question_column = 'Request - Text Request'
            answer_column = 'Request - Text Answer'
            id_col = 'Request - ID'
            # logger.debug(f"Using columns: Question={question_column}, Answer={answer_column}")

            # Directory for saving uploads
            upload_dir = "upload1"
            os.makedirs(upload_dir, exist_ok=True)
            # logger.debug(f"Upload directory ensured at: {upload_dir}")

            # Process each uploaded file
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                file_extension = os.path.splitext(file_name)[1].lower()
                # logger.debug(f"Processing file: {file_name} with extension: {file_extension}")

                # Ensure the file is Excel
                if file_extension not in ['.xls', '.xlsx']:
                    raise ValueError(f"Unsupported file type: {file_extension}")

                # Save file to the uploads directory
                excel_file_path = os.path.join(upload_dir, file_name.lower())
                with open(excel_file_path, "wb") as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)
                # logger.debug(f"File saved locally as Excel at: {excel_file_path}")

                try:
                    # Read the Excel file into a DataFrame
                    df = pd.read_excel(excel_file_path, engine='openpyxl')
                    # logger.debug(f"Excel file {file_name} successfully read into DataFrame")

                    # Process the file
                    process_file(
                        excel_file_path,
                        rag_system,
                        db_manager,
                        question_column,
                        answer_column,
                        id_col
                    )
                # logger.info(f"File {file_name} processed successfully")
                except Exception as e:
                    logger.error(f"Error processing file {file_name}: {str(e)}")

            return JsonResponse({"message": "Files processed successfully!"})
        except Exception as e:
            # logger.error(f"Error in processing_files API: {str(e)}")
            return JsonResponse({"error": str(e)}, status=500)
    else:
        logger.warning("Invalid HTTP method. Only POST is supported.")
        return JsonResponse({"error": "Invalid method. Only POST is allowed."}, status=405)


@csrf_exempt
def query_system(request):
    if request.method == 'POST':

        query = request.POST["sql_query"]
        api_key = os.getenv("OPENAI_API_KEY")
        # print(f"API Key: {api_key}")  # Debugging API key

        address = os.getenv("HANA_ADDRESS")
        # print(f"Address: {address}")  # Debugging HANA address
        port = os.getenv("HANA_PORT")
        # print(f"Port: {port}")  # Debugging HANA port
        user = os.getenv("HANA_USER")
        password = os.getenv("HANA_PASSWORD")

        qdrant_url = os.getenv("QDRANT_URL")
        # print(qdrant_url)
        qdrant_api = os.getenv("QDRANT_API")

        if not query:
            return JsonResponse({"error": "Query is required."}, status=400)

        # Initialize RAGSystem, DatabaseManager, and other components without sessions
        rag_system = RAGSystem(api_key, qdrant_url, qdrant_api)
        db_manager = DatabaseManager(address, port, user, password)

        memory = ConversationBufferWindowMemory(
            ai_prefix="Assistant",
            human_prefix="User",
            return_messages=True, memory_key='chat_history', input_key='input', k=5)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
        metadata = db_manager.get_metadata()
        # print(metadata)

        tools = [
            Tool(
                name="execute_sap_hana_query",
                func=lambda x: DatabaseManager(address, port, user, password).execute_query(
                    x.strip("'").rstrip("'")),
                description="Tool for executing SQL query on a SAP HANA database. The query should be enclosed in <sql></sql> tags. Use single quotes for strings and double quotes for table and columns names"
            ),
            Tool(
                name="query_RAG",
                func=rag_system.query_similar,
                description="Tool to retrieve similar text when a text closely matches previous entries in RAG system. The input text should be enclosed in <rag></rag> tags."
            )
        ]

        prompt = hub.pull("hwchase17/react-chat")

        prompt.template = """Assistant is a sophisticated language model developed by Digiotai Solutions, designed to assist with a wide array of tasks—from answering simple questions to providing detailed explanations and engaging in discussions on various topics. As a language model, Assistant generates human-like text based on the input it receives, enabling it to participate in natural-sounding conversations and provide coherent, relevant responses.

        Assistant is continually learning and improving, with capabilities that evolve over time. It can process and understand large volumes of text, using this knowledge to deliver accurate and informative responses to a broad range of queries. Additionally, Assistant can generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on numerous topics.

        Assistant is designed to answer user queries by leveraging a SQL database and a RAG system (a vector database for question similarity search). It is also equipped to handle unclear inputs, such as misspelled column names or values, ambiguous queries, or incomplete information, by leveraging the provided metadata and context.

        Assistant is designed to answer user queries by leveraging a SQL database and RAG sytem (a vector database for question similarity search). It always with responds with one of ('Thought', 'Action', 'Action Input', 'Observation', 'Final Answer')

        Also, Assistant is an expert data analyst and SQL query specialist, particularly skilled with SAP HANA databases. Assistant writes precise, efficient queries and provides clear, insightful analysis of the results.

        TOOLS:
        ------

        Assistant has access to the following tools:

        {tools}

        To use a tool, please use the following format:

        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (no additional text)
        Observation: the result of the action
        ```
        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        ```

        **Response Format**:  

        | Request - User Name                     | Number of Tickets |
        |-----------------------------------------|-------------------|
        | BROOKS MICHAEL                         | 15                |
        | BUSTOS ROMERO LILIANA ORFELINA        | 37                |
        | CREVIER RYAN                           | 3                 |
        | FRANCESCON CRISTINA                    | 17                |
        | FUSARI GIOVANNI                       | 3                 |
        | GONCALVES SERGIO REINALDO             | 5                 |
        | GUNNING GEORGE                        | 10                |
        | HOPEWELL NICOLE                       | 5                 |
        | LAZZARINI SILVANA                     | 7                 |
        | MATTEI MATTEO                         | 3                 |
        | MORTAGNA EMIL                         | 3                 |
        | OLDIGES CHARLES                       | 17                |
        | TESSARO MARTINA                       | 3                 |
        | VOICU ANDREEA NICOLETA              | 6                 |
        | WELCH MELISSA                         | 3                 |
        ```

        - When retrieving data from the SQL database or the RAG system as the above format, always present the results in a clear and structured **tabular format** within the table tags.  
        - The table should have proper column headers that match the context of the user's query.  
        - If applicable, align the format of numeric, text, and date/time data for clarity.  

        Begin! Remember to maintain this exact format for all interactions and focus on writing clean, error-free SQL queries. Make sure to provide a Final Answer to the user's question.

        Table Structure:
        ----------------

        | Column Name                           | Description                                                                 | Example                                                                 |
        |---------------------------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------|
        | Req. Creation Date                   | The date when the request was created.                                     | 3/21/2022                                                               |
        | Creation Time                        | The time when the request was created.                                     | 08:59:49                                                                |
        | Req. Creation Date - Year Week ISO   | The creation date represented in ISO year-week format.                    | 202212                                                                  |
        | Request - ID                         | The unique identifier for the request.                                    | A878369L                                                                |
        | Request - Priority Description       | The priority level of the request.                                        | P4 - Low                                                                |
        | Historical Status - Status From      | The previous status of the request before the last change.                | Awaiting external provider                                             |
        | Historical Status - Status To        | The current status of the request after the last change.                  | Work in progress                                                        |
        | Historical Status - Change Date      | The date when the status of the request was changed.                      | 20/03/2023                                                              |
        | Historical Status - Change Time      | The time when the status of the request was changed.                       | 145206                                                                  |
        | Macro Area - Name                    | The macro area or category to which the request belongs.                  | SAP Tr-Finance (FICO) SAP Retail (FMS1) NA                              |
        | Request - Resource Assigned To - GROUP SAP MD | The group to which the resource assigned to the request belongs.         | (Empty)                                                                |
        | Macro Area (SAP) - MA Area           | The macro area within SAP to which the request belongs.                   | (Empty)                                                                 |
        | Request - User Name                  | The name of the user who created the request.                              | FRANCESCON CRISTINA                                                     |
        | Request - Resource Assigned To - Name| The name of the resource assigned to the request.                         | Menon Nivhin                                                            |
        | Req. Type - Description EN           | The type of the request described in English.                             | Service Request                                                         |
        | Req. Status - Description            | The current status description of the request.                            | Closed                                                                  |
        | Req. Closing Date                    | The date when the request was closed.                                      | 9/12/2023                                                               |
        | Request - Text Request               | The detailed description of the request provided by the user.             | Enter a description of your request: : Good morning, ..we need to understand why the large majority of documents on the Retail side do not correctly pass from SAP to EDICOM (SII), as they are not available on the platform. The EDICOM (SII) platform automates the issuing and receiving of large volumes of invoices through an authorised channel with the Sistema di Interscambio - SdI and offers integration mechanisms with any available ERP. ..Please find attached Excel file with examples of documents in SAP that do not appear in EDICOM. Please consider that is a Fiscal compliance in Spain and we need to fix the problem...Thank you. ....System (RP1/ CP1/ MIM) : RP1..Brand : N.A...COUNTRY : Spain..SAP Site # : SII - ERP..SAP User ID : GiovanniF..Is it a RE-FX issue? : true |
        | Request - Text Answer                | The resolution or answer provided for the request.                        | Requirement: The connection to EDICOM has not been established and there are many pending statutory information to be sent to EDICOM.....Resolution:. This case was first received as an incident mentioning that some details are missing to be sent to EDICOM. On further investigation, it is found that despite haivng an interface created for sending the documents to EDICOM, there was no active connection. When this was discussed with the EDICOM technical team, it is further understood that the fileformats were not as per their requirements, therefore cinfirming that the intial project for this interface was not completed or seen through till the end......After several sessions with EDICOM to develop and test the file format, and working with internal technical teams to setup the infrastructure per EDICOM's needs, the file format is now live in production. Busienss has identified some old statutory documents to be sent to EDICOM, which were triggered manually by the Triage, and stopping the current jobs (Z_RP1_I_FIN_0620_SPAIN_SII under Control-M ticket A1276696L) to manually manage the documents being sent to EDICOM.. We have intimated the business of the stopped job so that they can ascenrain of they would need to restart the job for sending future documents to be sent to EDICOM, or not......There were several analysis performed to understand why specific documents failed to be sent to EDICOM, this was addressed in many email communications (most of them attached on this case)and suitable actions were taken by the Triage team......Now that there is currently no activity to be perfomed by Triage, and business is in the process of still reconciling the doucments to be sent, this service case would have to be considered as resolved. Any future requests to send stat documents to EDICOM can be addressed by creating new service requests spcifically for those set of documents while business can analyse further document that would need to be sent to EDICOM, then Triage willaddress them in the new service requests. |
        | Request - Subject description        | A brief description or subject of the request.                             | FIN_FINGLD_ES_XXX_XXXX - EDICOM does not send all documents to Sistema di Interscambio, FIN_FIN-AP_XX_XXX_XXXX - Activating fields Instruction Key 3 and 4 in the invoice registration posting - FMS, FIN_FINGLD_XX_TMV_XXX - Team Vision segment name change - A08, FIN_FIN-CONFIG_XX_XXX_XXXX - New Trading Partner Request, FIN_EYEMED_XX_XXX_XXXX: Discount transaction failing in LUW process, FIN_REFXPP_CL_GMO_5672 - Help with condition applied to contracts, FIN_EYEMED_US_XXX_E100 _Voids are entered into SAP and the pricing condition is not applied to the RP1, FIN_REFXVP_EC_GMO_6934 - Error in the contract 8139/5000041, FIN_FIN-AP_XX_XXX_XXXX : Positive Pay File Update for Company 1000, FIN_FIN-GL_XX_XXX_XXXX - Error during clearing of open items in GL 1141110683, FIN_FINTAX_IT_XXX_XXXX: Missing lines in S_P00_07000134 report, FIN_FIN-AR_XX_XXX_XXXX - Update remittance details appearing on customer statements using Tcode ZF_CST, FIN_POSDMX_CA_XXX_XXXX - POS Payments receipts are not clearing exactly with the customer invoices |

        Previous conversation history:
        {chat_history}

        Question: {input}
        {agent_scratchpad}
        """

        # chat_history = ""  # Initialize as empty if no conversation history exists

        # Create the agent and execute the query
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True,
                                       handle_parsing_errors=True)

        command = f"""
                                       Answer the queries from 'excel_data' tables:

                                       Metadata of the tables:
                                       {metadata}

                                       User query:
                                       {query}
                                   """

        # print(f"Command to agent: {command}")  # Debugging command sent to agent

        response = agent_executor.invoke({"input": command})

        return JsonResponse({"response": markdown_to_html(response['output'])})


# import markdown
# def markdown_to_html(md_text):
#     html_text = markdown.markdown(md_text)
#     return html_text

# import mistune
# def markdown_to_html(md_text):
#     markdown = mistune.create_markdown()
#     html = markdown(md_text)
#     print(html)
#     return html

from markdown_it import MarkdownIt


def markdown_to_html(md_text):
    md = MarkdownIt()
    html = md.render(md_text)
    print(html)
    return html