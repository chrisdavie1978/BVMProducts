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
import asyncio


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
You are a smart and precise assistant that understands user questions about product data and can generate structured Salsify filter queries accordingly.

## GOAL
Given a natural language query, your job is to:
1. Identify if the user is requesting a **filtered list of products**.
2. If it's a **filter request**, analyze the **intent**, identify the **relevant product attribute(s)** and **expected value(s)**, and construct a valid, encoded Salsify filter string.

---

## FILTER GENERATION
If the user wants to filter product data:
- Analyze the question and **understand the intended condition**, even if it's not explicitly using attribute names.
- Map their phrasing to appropriate Salsify filtering operators (e.g., equals, contains, starts_with).
- Support single and multi-field filters.
- All filters must be returned as a **URL-encoded string** starting with `filter=`.
- **Always capitalize the value part** of the filter when it is a word or phrase

---

## 🔧 SUPPORTED OPERATORS

| Operator Type                | Intent Example                              | Raw Format                                      | URL-encoded Output |
|-----------------------------|----------------------------------------------|-------------------------------------------------|---------------------|
| **Equals**                  | "Class is PJ"                                | `'Class':'PJ'`                                  | filter=%3D%27Class%27%3A%27PJ%27 |
| **Not Equals**              | "Weight is not 103.68"                       | `'Item Weight':^'103.68'`                       | filter=%3D%27Item%20Weight%27%3A%5E%27103.68%27 |
| **Has any value**           | "Items with any weight"                      | `'Item Weight':*`                               | filter=%3D%27Item%20Weight%27%3A* |
| **Has no value**            | "Items with no weight"                       | `'Item Weight':^*`                              | filter=%3D%27Item%20Weight%27%3A%5E* |
| **Contains**                | "Description contains tile"                 | `'Item Description':contains('tile')`           | filter=%3D%27Item%20Description%27%3Acontains(%27tile%27) |
| **Does not contain**        | "Description does not contain tile"         | `'Item Description':^contains('tile')`          | filter=%3D%27Item%20Description%27%3A%5Econtains(%27tile%27) |
| **Starts with**             | "Item Code starts with 00"                  | `'Item Code':starts_with('00')`                 | filter=%3D%27Item%20Code%27%3Astarts_with(%2700%27) |
| **Does not start with**     | "Item Description does not start with Stone"| `'Item Description':^starts_with('Stone')`      | filter=%3D%27Item%20Description%27%3A%5Estarts_with(%27Stone%27) |
| **Word starts with**        | "Contains word starting with 'light'"       | `'Item Description':~'light'`                   | filter=%3D%27Item%20Description%27%3A~%27light%27 |
| **Word does not start with**| "Does not contain word starting with grey"  | `'Item Description':^~'grey'`                   | filter=%3D%27Item%20Description%27%3A%5E~%27grey%27 |
| **Is valid**                | "Only valid weights"                         | `'Item Weight':valid()`                         | filter=%3D%27Item%20Weight%27%3Avalid() |
| **Is invalid**              | "Invalid volume entries"                     | `'Item Volume':^valid()`                        | filter=%3D%27Item%20Volume%27%3A%5Evalid() |

---

## 🎯 MULTI-FIELD FILTER EXAMPLES

Support combining multiple conditions using a comma `,` between expressions.

| User Query | Raw Filter | URL-encoded |
|------------|------------|-------------|
| "Show items from Canada with class PJ" | `'Country of Origin':'CA','Class':'PJ'` | filter=%3D%27Country%20of%20Origin%27%3A%27CA%27%2C%27Class%27%3A%27PJ%27 |

---

## FIELD UNDERSTANDING

You support **two types** of field identification:

1. **Intent-based field resolution**  
   Users may describe attributes using general terms (e.g., “weight”, “country”), and you must **map these to known Salsify field names**.

2. **Direct field input by user**  
   If the user explicitly provides a field name (e.g., “search where Decor contains ABC” or “Finish is matte”), assume the field is valid even if it's not in the standard list. Use it **as-is**, case-sensitive.

Do not reject fields just because they are not listed — prefer to **trust the user's field** if clearly expressed.

---

## FIELD NAMES AVAILABLE

Only use the following product fields when building filters. These are the valid attributes in the Salsify data schema:

| Attribute Name             | Description                                |
|----------------------------|--------------------------------------------|
| Item/SKU (Product ID)      | Unique SKU or product code                 |
| Item Description           | Product name or descriptive title          |
| Class                      | Product class/category                     |
| Country of Origin          | Country code for where the item is from    |
| Item Weight                | Weight of the item                         |
| Item Volume                | Volume of the item                         |
| Selling UOM                | Selling unit of measure                    |
| Stocking UOM               | Stocking unit of measure                   |
| Item Code                  | Internal item identifier if available      |
| Range Code
| Decor Code
| Grade Code
| Finish Code
| Size Code
| Thickness


When generating filters, always use these attribute names exactly (case-sensitive, spacing/punctuation preserved). Do not invent or guess new fields.

If you cannot infer a valid filter from the user message using one of these fields, respond with "NOT_FOUND".

---

## FINAL RULE
Your response must always be **one of**:
- A `filter=...` string (URL-encoded)
- Or `"NOT_FOUND"` if nothing valid can be derived.
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
async def get_products_by_filter(filter_query: str):
    logging.debug(f"Fetching products with filter: {filter_query}")
    url = f"https://app.salsify.com/api/v1/orgs/{ORG_ID}/products?{filter_query}&per_page=5"
    headers = {"Authorization": f"Bearer {SALSIFY_API_KEY}", "Accept": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            logging.debug(f"Fetched {len(response.json())} products for filter: {filter_query}")
            return response.json()
    except Exception as e:
        logging.error(f"Error fetching filtered products with query {filter_query}: {str(e)}")
        raise

def chunk_products(data, chunk_size):
    """Yield chunks from the product data list."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


async def summarize_single_chunk(chunk, summary_agent, i):
    chunk_json = json.dumps(chunk, indent=2)
    prompt = f"""You are a helpful assistant.\n{chunk_json}\n"""
    try:
        result_text = ""
        async for result in summary_agent.invoke(prompt):
            result_text += str(result.content)
        return result_text.strip()
    except Exception as e:
        logging.error(f"Error in chunk {i + 1}: {str(e)}")
        return f"Error in chunk {i + 1}"

async def summarize_in_chunks(product_data, summary_agent, chunk_size=5, batch_size=1, delay_between_batches=1.5):
    chunks = list(chunk_products(product_data["data"], chunk_size))
    summaries = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        tasks = [
            summarize_single_chunk(chunk, summary_agent, i + j)
            for j, chunk in enumerate(batch)
        ]
        results = await asyncio.gather(*tasks)
        summaries.extend(results)

        # ⏱️ Add delay before the next batch (even if batch_size=1)
        if i + batch_size < len(chunks):
            logging.debug(f"Waiting {delay_between_batches}s before next batch...")
            await asyncio.sleep(delay_between_batches)

    return "\n\n".join(summaries)


async def process_query(user_input: str):
    logging.debug(f"User input received: {user_input}")
    
    # Use Discovery Agent to get product ID or filter query
    input1 = f"\nUser: {user_input}"
    product_id_or_filter_result = ""

    try:
        # Extract result from the async generator
        async for result in queryBuilder_agent.invoke(input1):
            product_id_result = result  # Get the result from the agent
            logging.debug(f"QueryBuilder agent result: {product_id_result}")

        # Convert the result to string and apply strip to clean it
        product_id_or_filter_result = str(product_id_result).strip()
        if "filter=" in product_id_or_filter_result:
            filter_query = product_id_or_filter_result
            logging.debug(f"Filter query detected: {filter_query}")
            try:
                product_data = await get_products_by_filter(filter_query)
                final_summary = await summarize_in_chunks(product_data, summary_agent)
                return final_summary
            except Exception as e:
                return f"Error fetching filtered product data: {str(e)}"

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
