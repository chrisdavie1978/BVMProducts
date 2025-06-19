import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json, re, httpx, asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent


# Load environment variables from .env file
load_dotenv()

# === Azure OpenAI & Salsify config ===
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
SALSIFY_API_KEY = os.getenv("SALSIFY_API_KEY")
ORG_ID = os.getenv("ORG_ID")

# === System prompts for agents ===
QUERY_BUILDER_SYSTEM_PROMPT = """
You are a precise assistant who understands product-related user queries.

If the input asks about a product directly (e.g. "Tell me about product 006921058710000"), extract the product ID.
- Product ID is a 12+ character alphanumeric string. Respond with only the ID.

If nothing is matched, return 'NOT_FOUND'.
"""

SUMMARY_SYSTEM_PROMPT = """
You are a helpful assistant that summarizes product data from JSON.
"""

# === Kernel and agents setup ===
kernel = Kernel()
kernel.add_service(AzureChatCompletion(
    deployment_name=AZURE_DEPLOYMENT_NAME,
    endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_API_VERSION
))

queryBuilder_agent = ChatCompletionAgent(
    kernel=kernel,
    name="product-QueryBuilder-agent",
    instructions=QUERY_BUILDER_SYSTEM_PROMPT
)

summary_agent = ChatCompletionAgent(
    kernel=kernel,
    name="summary-agent",
    instructions=SUMMARY_SYSTEM_PROMPT
)

# === FastAPI setup ===
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
session_memory = []

# === Setup Logging ===
logging.basicConfig(level=logging.DEBUG)

# === Helpers ===
async def get_product_data(product_id: str):
    logging.debug(f"Fetching data for product ID: {product_id}")
    url = f"https://app.salsify.com/api/v1/orgs/{ORG_ID}/products/{product_id}"
    headers = {"Authorization": f"Bearer {SALSIFY_API_KEY}", "Accept": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            logging.debug(f"Product data fetched successfully for {product_id}")
            return response.json()
    except Exception as e:
        logging.error(f"Error fetching product data for {product_id}: {str(e)}")
        raise

async def process_query(user_input: str):
    logging.debug(f"User input received: {user_input}")
    
    # Use Discovery Agent to get product ID or filter query
    input1 = f"{QUERY_BUILDER_SYSTEM_PROMPT}\n\nUser: {user_input}"
    product_id_or_filter_result = ""

    try:
        # Extract result from the async generator
        async for result in queryBuilder_agent.invoke(input1):
            product_id_result = result  # Get the result from the agent
            logging.debug(f"QueryBuilder agent result: {product_id_result}")

        # Convert the result to string and apply strip to clean it
        product_id_or_filter_result = str(product_id_result).strip()

        # Case 1: Product ID found
        if re.match(r"^[A-Za-z0-9]{12,}$", product_id_or_filter_result):
            product_id = product_id_or_filter_result
            logging.debug(f"Product ID detected: {product_id}")
            try:
                product_data = await get_product_data(product_id)
                session_memory.append(product_id)
                logging.debug(f"Session memory updated with product ID: {product_id}")
            except Exception as e:
                return f"Error fetching product data: {str(e)}"

            raw_json = json.dumps(product_data, indent=2)
            summary_input = f"{SUMMARY_SYSTEM_PROMPT}\n\nHere is the full product data:\n{raw_json}"
            async for result2 in summary_agent.invoke(summary_input):
                logging.debug(f"Summary agent response: {result2.content}")
                return str(result2.content)

        # Case 3: Nothing valid
        else:
            logging.error("No valid product ID or filter query found in the input.")
            return "No valid product ID or filter query found in the input."
    except Exception as e:
        logging.error(f"Error during query processing: {str(e)}")
        return f"Error during query processing: {str(e)}"

# === FastAPI endpoints ===
@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("chatbot_interface.html", {"request": request, "memory": session_memory})

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    user_input = body.get("message", "")
    logging.debug(f"Processing chat message: {user_input}")
    reply = await process_query(user_input)
    logging.debug(f"Reply generated: {reply}")
    return {"reply": reply}
