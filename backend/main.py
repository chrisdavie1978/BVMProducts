# main.py
import os
import json
import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatHistory, FunctionCallContent
import re

# Import configurations, prompts, and plugins
from config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_DEPLOYMENT_NAME,
    AZURE_API_VERSION, SALSIFY_API_KEY, ORG_ID
)
from prompts import CONVERSATIONAL_AGENT_SYSTEM_PROMPT # Only need this one now
from plugins import SalsifyProductTool # Import the Salsify tool class

# The basic logging setup is now in config.py and already imported/configured.


# === Kernel and Agent Initialization ===
kernel = Kernel()

# Using the requested format: inline instantiation of AzureChatCompletion
kernel.add_service(
    AzureChatCompletion(
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        deployment_name=AZURE_DEPLOYMENT_NAME,
        api_version=AZURE_API_VERSION,
    )
)

# Instantiate and register the Salsify tool
salsify_tool = SalsifyProductTool(SALSIFY_API_KEY, ORG_ID)
kernel.add_plugin(salsify_tool, plugin_name="SalsifyProductTool")

# === Main Conversational Agent (now handles summarization directly) ===
conversationalAgent = ChatCompletionAgent(
    kernel=kernel,
    name="conversational-salsify-assistant",
    instructions=CONVERSATIONAL_AGENT_SYSTEM_PROMPT
)

# === FastAPI Setup ===
app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# Session memory to store chat history
session_memory = {}

# === Helper Functions ===

async def process_query(user_input: str, session_id: str = "default"):
    """
    Main function to process user queries using a single conversational agent,
    which now automatically handles tool calls (Salsify API) and then summarization internally.
    Language inference is handled by the agent based on conversation history.
    """
    logging.info(f"Processing user query: '{user_input}' for session: {session_id}")

    # Initialize session memory
    if session_id not in session_memory:
        session_memory[session_id] = {
            "chat_history": ChatHistory(),
        }
        # Add system prompt upon initialization
        session_memory[session_id]["chat_history"].add_system_message(CONVERSATIONAL_AGENT_SYSTEM_PROMPT)

    current_chat_history = session_memory[session_id]["chat_history"]

    # Add user input to history, unless it's the initial empty trigger
    if user_input:
        current_chat_history.add_user_message(user_input)

    logging.info(f"Invoking Conversational Agent for session: {session_id}")

    final_agent_response_content = ""
    try:
        async for agent_response in conversationalAgent.invoke(current_chat_history):
            logging.info(f"Conversational Agent raw step response: {agent_response}")
            if agent_response.content:
                final_agent_response_content = agent_response.content

            if agent_response.metadata and agent_response.metadata.get('usage'):
                logging.info(f"Conversational Agent Token Usage: {agent_response.metadata.get('usage')}")

    except Exception as e:
        logging.error(f"Error during Conversational Agent invocation: {e}", exc_info=True)
        error_message = "I encountered an issue trying to process your request. Please try rephrasing."
        current_chat_history.add_assistant_message(error_message)
        return {"reply": error_message}

    response_to_send = str(final_agent_response_content).strip()
    logging.info(f"Conversational Agent's final response to user: '{response_to_send}'")

    current_chat_history.add_assistant_message(response_to_send)

    return {"reply": response_to_send}


# === FastAPI Routes ===

# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     """
#     Serves the main HTML page for the chatbot interface.
#     """
#     return templates.TemplateResponse("chatbot_interface.html", {"request": request})

@app.post("/chat")
async def chat(request: Request):
    """
    Handles incoming chat messages from the frontend.
    """
    try:
        body = await request.json()
        user_input = body.get("message", "").strip()
        session_id = body.get("session_id", "default")

        # If it's the very first request (empty input from onload), process it to get initial greeting
        if not user_input and (session_id not in session_memory or not session_memory[session_id]["chat_history"].messages):
             return await process_query("", session_id) # Send empty message to trigger initial greeting logic
        elif not user_input: # If input is empty but session is ongoing, prompt for message
             return {"reply": "Please enter a message."}

        reply = await process_query(user_input, session_id)
        return reply
    except json.JSONDecodeError:
        logging.error("Invalid JSON received in /chat endpoint.")
        return {"reply": "Invalid request format."}
    except Exception as e:
        logging.error("Error during /chat endpoint processing:", exc_info=True)
        return {"reply": "An unexpected error occurred while processing your request. Please try again."}