# from fastapi import FastAPI
# from pydantic import BaseModel
# import requests
# from bs4 import BeautifulSoup
# #from textblob import TextBlob
# import re
# import os
#
# app = FastAPI()
#
# class QueryRequest(BaseModel):
#     prompt: str
#
# SERPAPI_KEY = "8c8faa817b1ab0b31f2cb0fffe275c71671519099f0b6a4ba9f84d7b4434fd7a"
#
# def google_search(query: str):
#     search_url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
#     response = requests.get(search_url)
#     if response.status_code == 200:
#         data = response.json()
#         return [result["link"] for result in data.get("organic_results", [])[:2]]
#     return []
#
# def scrape_content(url):
#     headers = {"User-Agent": "Mozilla/5.0"}
#     try:
#         response = requests.get(url, headers=headers, timeout=5)
#         soup = BeautifulSoup(response.text, "html.parser")
#         paragraphs = soup.find_all("p")
#         text = " ".join([para.get_text() for para in paragraphs])
#         return re.sub(r'\s+', ' ', text).strip()
#     except Exception:
#         return ""
#
# # def analyze_text(text: str):
# #     blob = TextBlob(text)
# #     sentiment = blob.sentiment.polarity
# #     keywords = list(set(blob.words))[:10]  # Extract top 10 unique words
# #     summary = " ".join(text.split()[:50]) + "..."  # Shortened summary
# #     return {"sentiment": sentiment, "keywords": keywords, "summary": summary}
#
# @app.post("/analyze")
# def analyze_query(request: QueryRequest):
#     urls = google_search(request.prompt)
#     print(urls)
#     content = " ".join([scrape_content(url) for url in urls])
#     if not content:
#         return {"error": "No relevant content found."}
#     # analysis = analyze_text(content)
#     print(content)
#     return {"query": request.prompt, "Content":content}
#
#


import base64
import json
from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
import requests
from dotenv import  load_dotenv
load_dotenv()

app = FastAPI()


class QueryRequest(BaseModel):
    prompt: str


prompt1 = """
You are an expert financial analyst providing real-time insights into company performance. Analyze the financial data of [Company Name]  using the latest available information from reliable sources, including MarketScreener, Bloomberg, Mint, Wikipedia, and company filings.
Ensure the response follows this JSON format:
{
  "title": "[Suitable title with Company Name]",
  "summary": "Brief overview of financial performance of [Company Name]",
  "paragraphs": [
    {
      "content": "Detailed analysis including revenue,statistics, net profit, EPS, and growth of [Company Name]"
    },
    {
      "content": "Comparison with competitors and market trends of [Company Name]."
    },
    {
      "content": "Predicted growth areas and risks based on data ."
    }
  ]
}

Requirements:
-Do not include any '**' or '##" or anything in the final response.just give the plain text as the response.
-'Summary' should be present for 3-4 lines
-'Context' should be within 1-2 lines and the very important content should be there.
-'title' should be inserted based on the user query only.The title must be unique.
- All the content and the summary must and should collect from the Sources such as "Wikipedia" or "Company Press releases" or "each metric from reliable data" only.
"""


def query_openai(prompt: str):
    """Calls OpenAI API using the updated method."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt1},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=1200
    )

    raw_response = response.choices[0].message.content.strip()

    try:
        structured_data = json.loads(raw_response)
    except json.JSONDecodeError:
        return {"error": "Failed to parse response as JSON."}

    return structured_data


@app.post("/analyze")
def analyze_query(request: QueryRequest):
    """Processes user query and returns structured response."""
    if not request.prompt:
        return {"error": "Query cannot be empty."}

    analysis = query_openai(request.prompt)

    return {"query": request.prompt, "response": analysis}



#Image Generation api starts
class ImageRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"  # Default image size


def generate_image(prompt: str, size: str):
    """Calls OpenAI's DALL·E 2 model to generate an image based on the prompt."""
    valid_sizes = ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]
    if size not in valid_sizes:
        size = "1024x1024"  # Fallback to a default valid size

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.images.generate(
        model="dall-e-2",
        prompt=f"Animated illustration of images with vibrant colors, cartoon-style characters, and a lively atmosphere for {prompt}",
        n=1,  # Number of images to generate
        size=size
    )
    image_url = response.data[0].url
    return image_url


def convert_image_to_base64(image_url: str):
    """Downloads the image and converts it to base64 encoding."""
    response = requests.get(image_url)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    return None


@app.post("/generate-image")
def generate_image_api(request: ImageRequest):
    """Processes user prompt and returns an AI-generated image in base64 format."""
    try:
        image_url = generate_image(request.prompt, request.size)
        print(image_url)
        image_base64 = convert_image_to_base64(image_url)
        if image_base64:
            return {"query": request.prompt, "image_base64": image_base64}
        return {"error": "Failed to convert image to base64."}
    except Exception as e:
        return {"error": str(e)}
