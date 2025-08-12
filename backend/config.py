# config.py
import os
import logging
from dotenv import load_dotenv
from datetime import datetime # For prompt time

load_dotenv() # Load environment variables from .env file

# --- Environment Variables ---
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
SALSIFY_API_KEY = os.getenv("SALSIFY_API_KEY")
ORG_ID = os.getenv("ORG_ID")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Debugging: Print loaded environment variables ---
logging.info(f"DEBUG: AZURE_OPENAI_ENDPOINT loaded: '{AZURE_OPENAI_ENDPOINT}'")
logging.info(f"DEBUG: AZURE_DEPLOYMENT_NAME loaded: '{AZURE_DEPLOYMENT_NAME}'")
logging.info(f"DEBUG: AZURE_API_VERSION loaded: '{AZURE_API_VERSION}'")
if not AZURE_DEPLOYMENT_NAME:
    logging.error("ERROR: AZURE_OPENAI_DEPLOYMENT_NAME is empty or None. Please check your .env file.")
    exit(1) # Exit if critical environment variable is missing

# --- Global Data/Settings (if any) ---
# You can add global configurations here, e.g., if you want to limit products summarized
MAX_PRODUCTS_TO_SUMMARIZE = os.getenv("MAX_PRODUCTS_TO_SUMMARIZE")
MAX_SUMMARY_WORD_COUNT = os.getenv("MAX_SUMMARY_WORD_COUNT") # Per product summary part