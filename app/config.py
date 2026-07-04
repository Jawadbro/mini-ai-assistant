"""
Central configuration for the mini AI assistant.

All tunables live here so the rest of the codebase never hardcodes
model names, paths, or thresholds. Values are overridable via .env.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
ORDERS_FILE = DATA_DIR / "orders.json"
PRODUCTS_FILE = DATA_DIR / "products.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# --- OpenAI --------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# Must match the output dimension of EMBEDDING_MODEL above.
# text-embedding-3-small -> 1536, text-embedding-3-large -> 3072
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1536))

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

# --- Retrieval --------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", 4))
# FAISS returns cosine similarity in [-1, 1] (via normalized inner product).
# Chunks scoring below this are considered "not relevant enough" to answer from.
MIN_RELEVANT_SIMILARITY = float(os.getenv("MIN_RELEVANT_SIMILARITY", 0.25))

# --- Memory --------------------------------------------------------------
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 10))

NO_ANSWER_MESSAGE = "I couldn't find that information in the uploaded documents."
