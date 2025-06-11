from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="BVM Chatbot",
    description="A simple chatbot using OpenAI. Enables asking questions and getting answers based on uploaded documents.",
    version="0.1"
)

SALSIFY_API_TOKEN = os.getenv("SALSIFY_API_TOKEN")
SALSIFY_DOMAIN = os.getenv("SALSIFY_DOMAIN")

class Query(BaseModel):
    message: str

@app.post("/chat")
async def chat(query: Query):
    product_name = query.message.strip()

    headers = {
        "Authorization": f"Bearer {SALSIFY_API_TOKEN}"
    }

    url = f"https://{SALSIFY_DOMAIN}/api/products?filter=name:{product_name}"

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return {"Product not identified"}

    try:
        data = response.json()
    except ValueError as e:
        return {"response": f"Failed to parse JSON: {str(e)}", "raw": response.text}

    if not data.get("products"):
        return {"response": f'No product found with name "{product_name}".'}

    product = data["products"][0]
    name = product.get("name", "N/A")
    price = product.get("price", "N/A")
    description = product.get("description", "N/A")

    return {
        "response": f"**Product:** {name}\n**Price:** {price}\n**Description:** {description}"
    }

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the chatbot interface at the root URL
@app.get("/")
async def root():
    return FileResponse("static/chatbot_interface.html")
