import psycopg2

class PostgreSQLDB:
    def __init__(self, dbname, user, password, host='ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech', port=5432):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def connect(self):
        try:
            conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            return conn
        except Exception as e:
            print(e)
            return None

    #Environmment Table creation
    # Environment Table creation with removed columns
    def table_creation(self):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = """
                CREATE TABLE IF NOT EXISTS environment (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    api_key TEXT NOT NULL,
                    model VARCHAR(100) NOT NULL,
                    temperature NUMERIC(3, 2) DEFAULT 0.5,
                    email VARCHAR(255) NOT NULL
                );
                """
                cursor.execute(query)
                conn.commit()
                cursor.close()
                conn.close()
        except Exception as e:
            print(f"Error creating environment table: {e}")

    # Table deletion for environment
    def table_deletion(self):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "DROP TABLE IF EXISTS environment CASCADE;"
                cursor.execute(query)
                conn.commit()
                cursor.close()
                conn.close()
        except Exception as e:
            print(f"Error deleting environment table: {e}")

    # Create Environment without top_p and model_vendor
    def create_environment(self, name, api_key, model, temperature=0.5, email=None):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = """
                INSERT INTO environment (name, api_key, model, temperature, email)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """
                cursor.execute(query, (name, api_key, model, temperature, email))
                environment_id = cursor.fetchone()[0]
                conn.commit()
                cursor.close()
                conn.close()
                print(f"Environment added with ID: {environment_id}")
                return environment_id
        except Exception as e:
            print(f"Error creating environment: {e}")
            return None

    # Read environment by ID
    def read_environment(self, environment_id):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "SELECT * FROM environment WHERE id = %s;"
                cursor.execute(query, (environment_id,))
                environment = cursor.fetchone()
                cursor.close()
                conn.close()
                if environment:
                    print(f"Environment found: {environment}")
                    return environment
                else:
                    print(f"No environment found with ID: {environment_id}")
                    return None
        except Exception as e:
            print(f"Error reading environment: {e}")
            return None

    # Update environment with email field
    def update_environment(self, environment_id, name=None, api_key=None, model=None, temperature=None, email=None):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = """
                UPDATE environment
                SET name = COALESCE(%s, name),
                    api_key = COALESCE(%s, api_key),
                    model = COALESCE(%s, model),
                    temperature = COALESCE(%s, temperature),
                    email = COALESCE(%s, email)
                WHERE id = %s;
                """
                cursor.execute(query, (name, api_key, model, temperature, email, environment_id))
                conn.commit()
                cursor.close()
                conn.close()
                print(f"Environment with ID {environment_id} updated.")
        except Exception as e:
            print(f"Error updating environment: {e}")

    # Read all environments with email field
    def read_all_environments(self):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "SELECT * FROM environment;"
                cursor.execute(query)
                environments = cursor.fetchall()
                cursor.close()
                conn.close()
                if environments:
                    for environment in environments:
                        print(
                            f"ID: {environment[0]}, Name: {environment[1]}, Model: {environment[3]}, Temperature: {environment[4]}, Email: {environment[5]}")
                    return environments
                else:
                    print("No environments found.")
                    return None
        except Exception as e:
            print(f"Error reading all environments: {e}")
            return None

    # Delete environment by ID
    def delete_environment(self, environment_id):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "DELETE FROM environment WHERE id = %s;"
                cursor.execute(query, (environment_id,))
                conn.commit()
                cursor.close()
                conn.close()
                print(f"Environment with ID {environment_id} deleted.")
        except Exception as e:
            print(f"Error deleting environment: {e}")

    def get_environments_by_email(self, email):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "SELECT * FROM environment WHERE email = %s;"
                cursor.execute(query, (email,))
                environments = cursor.fetchall()
                cursor.close()
                conn.close()
                return environments
        except Exception as e:
            print(f"Error fetching environments by email: {e}")
            return None



    #Agents table
    # Create agents table linked with environments, including 'tools' column
    # Updated Code with Swapped Positions for 'backend_id' and 'tools'
    def create_agents_table(self):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = """
                CREATE TABLE IF NOT EXISTS ai_all_agents (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    system_prompt TEXT,
                    agent_description TEXT,
                    backend_id TEXT,  
                    tools TEXT,  
                    upload_attachment BOOLEAN DEFAULT FALSE,  
                    env_id INT REFERENCES environment(id) ON DELETE CASCADE,
                    dynamic_agent_id INT REFERENCES dynamic_ai_agents(id) ON DELETE CASCADE,
                    email VARCHAR(255),
                    image_id INT  -- New column added
                );
                """
                cursor.execute(query)
                conn.commit()
                cursor.close()
                conn.close()
                print("Agents table created successfully with image_id column.")
        except Exception as e:
            print(f"Error creating agents table: {e}")

    def drop_agents_table(self):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "DROP TABLE IF EXISTS ai_all_agents;"
                cursor.execute(query)
                conn.commit()
                cursor.close()
                conn.close()
                print("Agents table deleted.")
        except Exception as e:
            print(f"Error deleting agents table: {e}")

    def create_agent(self, name, system_prompt, agent_description, backend_id, tools, upload_attachment, env_id,
                     dynamic_agent_id, email, image_id):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = """
                INSERT INTO ai_all_agents 
                (name, system_prompt, agent_description, backend_id, tools, upload_attachment, env_id, dynamic_agent_id, email, image_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                cursor.execute(query, (
                    name, system_prompt, agent_description, backend_id, tools, upload_attachment, env_id,
                    dynamic_agent_id, email, image_id
                ))
                agent_id = cursor.fetchone()[0]
                conn.commit()
                cursor.close()
                conn.close()
                return agent_id
        except Exception as e:
            print(f"Error creating agent: {e}")
            return None

    def read_agent(self, agent_id):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "SELECT * FROM ai_all_agents WHERE id = %s;"
                cursor.execute(query, (agent_id,))
                agent = cursor.fetchone()
                cursor.close()
                conn.close()
                return agent
        except Exception as e:
            print(f"Error reading agent: {e}")
            return None

    def update_agent(self, agent_id, name=None, system_prompt=None, agent_description=None, backend_id=None,
                     tools=None, upload_attachment=None, env_id=None, dynamic_agent_id=None, email=None, image_id=None):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = """
                UPDATE ai_all_agents
                SET name = COALESCE(%s, name),
                    system_prompt = COALESCE(%s, system_prompt),
                    agent_description = COALESCE(%s, agent_description),
                    backend_id = COALESCE(%s, backend_id),
                    tools = COALESCE(%s, tools),
                    upload_attachment = COALESCE(%s, upload_attachment),
                    env_id = COALESCE(%s, env_id),
                    dynamic_agent_id = COALESCE(%s, dynamic_agent_id),
                    email = COALESCE(%s, email),
                    image_id = COALESCE(%s, image_id)  -- New column update
                WHERE id = %s;
                """
                cursor.execute(query, (
                    name, system_prompt, agent_description, backend_id, tools, upload_attachment, env_id,
                    dynamic_agent_id, email, image_id, agent_id
                ))
                conn.commit()
                cursor.close()
                conn.close()
                print(f"Agent with ID {agent_id} updated.")
        except Exception as e:
            print(f"Error updating agent: {e}")

    def delete_agent(self, agent_id):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "DELETE FROM ai_all_agents WHERE id = %s;"
                cursor.execute(query, (agent_id,))
                conn.commit()
                cursor.close()
                conn.close()
                print(f"Agent with ID {agent_id} deleted.")
        except Exception as e:
            print(f"Error deleting agent: {e}")

    def get_all_agents(self):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = """
                SELECT id, name, system_prompt, agent_description, backend_id, tools, upload_attachment, env_id, dynamic_agent_id, email, image_id
                FROM ai_all_agents;
                """
                cursor.execute(query)
                agents = cursor.fetchall()
                cursor.close()
                conn.close()
                return agents
        except Exception as e:
            print(f"Error retrieving agents: {e}")
            return None

    def get_agents_by_email(self, email):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "SELECT * FROM ai_all_agents WHERE email = %s;"
                cursor.execute(query, (email,))
                agents = cursor.fetchall()
                cursor.close()
                conn.close()
                return agents
        except Exception as e:
            print(f"Error fetching agents by email: {e}")
            return None

    #  Dynamic Agents table
    def create_dynamic_agents_table(self):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = """
                CREATE TABLE IF NOT EXISTS dynamic_ai_agents (
                    id SERIAL PRIMARY KEY,
                    agent_name VARCHAR(100) NOT NULL,
                    agent_goal TEXT,
                    agent_description TEXT,
                    agent_instruction TEXT,
                    email VARCHAR(255)  -- Added email as the last column
                );
                """
                cursor.execute(query)
                conn.commit()
                cursor.close()
                conn.close()
                print("Dynamic_Agents table created successfully.")
        except Exception as e:
            print(f"Error creating agents table: {e}")

    # Drop dynamic_agents table
    def drop_dynamic_agents_table(self):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "DROP TABLE IF EXISTS dynamic_ai_agents;"
                cursor.execute(query)
                conn.commit()
                cursor.close()
                conn.close()
                print("Dynamic Agents table deleted.")
        except Exception as e:
            print(f"Error deleting agents table: {e}")

    # Insert a new agent
    def create_dynamic_agent(self, agent_name, agent_goal, agent_description, agent_instruction, email):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = """
                INSERT INTO dynamic_ai_agents (agent_name, agent_goal, agent_description, agent_instruction, email)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
                """
                cursor.execute(query, (agent_name, agent_goal, agent_description, agent_instruction, email))
                agent_id = cursor.fetchone()[0]  # Expecting a single row with the new agent ID
                conn.commit()
                cursor.close()
                conn.close()
                return agent_id
        except Exception as e:
            print(f"Error creating agent: {e}")
            return None

    # Read agent by ID
    def read_dynamic_agent(self, agent_id):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "SELECT * FROM dynamic_ai_agents WHERE id = %s;"
                cursor.execute(query, (agent_id,))
                agent = cursor.fetchone()
                cursor.close()
                conn.close()
                return agent
        except Exception as e:
            print(f"Error reading agent: {e}")
            return None

    # Update agent by ID
    def update_dynamic_agent(self, agent_id, agent_name=None, agent_goal=None, agent_description=None,
                             agent_instruction=None, email=None):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = """
                UPDATE dynamic_ai_agents
                SET agent_name = COALESCE(%s, agent_name),
                    agent_goal = COALESCE(%s, agent_goal),
                    agent_description = COALESCE(%s, agent_description),
                    agent_instruction = COALESCE(%s, agent_instruction),
                    email = COALESCE(%s, email)
                WHERE id = %s;
                """
                cursor.execute(query, (
                    agent_name, agent_goal, agent_description, agent_instruction, email, agent_id))
                conn.commit()
                cursor.close()
                conn.close()
                print(f"Agent with ID {agent_id} updated.")
        except Exception as e:
            print(f"Error updating agent: {e}")

    # Delete agent by ID
    def delete_dynamic_agent(self, agent_id):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "DELETE FROM dynamic_ai_agents WHERE id = %s;"
                cursor.execute(query, (agent_id,))
                conn.commit()
                cursor.close()
                conn.close()
                print(f"Agent with ID {agent_id} deleted.")
        except Exception as e:
            print(f"Error deleting agent: {e}")

    # Get all agents
    def get_all_dynamic_agents(self):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = """
                SELECT id, agent_name, agent_goal, agent_description, agent_instruction, email
                FROM dynamic_ai_agents;
                """
                cursor.execute(query)
                agents = cursor.fetchall()
                cursor.close()
                conn.close()
                return agents
        except Exception as e:
            print(f"Error retrieving agents: {e}")
            return None

    def get_dynamic_agents_by_email(self, email):
        try:
            conn = self.connect()
            if conn is not None:
                cursor = conn.cursor()
                query = "SELECT * FROM dynamic_ai_agents WHERE email = %s;"
                cursor.execute(query, (email,))
                dynamic_agents = cursor.fetchall()
                cursor.close()
                conn.close()
                return dynamic_agents
        except Exception as e:
            print(f"Error fetching dynamic agents by email: {e}")
            return None

if __name__ == "__main__":
    db = PostgreSQLDB(dbname='test', user='test_owner', password='tcWI7unQ6REA')
    db.table_creation()
    db.create_dynamic_agents_table()
    db.create_agents_table()
    #db.read_environment(1)
    # db.table_deletion()
    # db.drop_agents_table()
    # db.drop_dynamic_agents_table()

