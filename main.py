from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import openai

# Load environment variables
SALISFY_API_URL = os.getenv('SALISFY_API_URL')
SALISFY_API_KEY = os.getenv('SALISFY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

app = FastAPI()

# Request model
class ChatRequest(BaseModel):
    user_message: str
    product_id: str

def get_product_attributes(product_id):
    headers = {
        'Authorization': f'Bearer {SALISFY_API_KEY}',
        'Content-Type': 'application/json'
    }
    url = f"{SALISFY_API_URL}/products/{product_id}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch product data: {response.status_code} - {response.text}")

def generate_chatbot_response(user_message, product_attributes):
    # Convert product attributes dict to string
    attributes_str = "\n".join([f"{k}: {v}" for k, v in product_attributes.items()])

    prompt = (
        f"You are a helpful assistant. A user asked: '{user_message}'.\n"
        f"Product attributes are:\n{attributes_str}\n"
        "Provide a helpful, concise answer."
    )

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        product_data = get_product_attributes(request.product_id)
        response_text = generate_chatbot_response(request.user_message, product_data)
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
