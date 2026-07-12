"""
Central configuration for the RAG chatbot.
All values can be overridden via environment variables (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM provider settings ---
# Supported: "anthropic", "openai", "none" (retrieval-only demo mode, no API key needed)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "none").lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Retrieval settings ---
CHUNK_SIZE_WORDS = 120       # words per chunk
CHUNK_OVERLAP_WORDS = 30     # overlap between consecutive chunks
TOP_K = 3                    # number of chunks retrieved per query
MIN_RELEVANCE_SCORE = 0.08   # below this, we treat the query as "not found in docs"
                              # (TF-IDF cosine similarity scores are typically 0-1)

# --- Data paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOMER_DOCS_DIR = os.path.join(BASE_DIR, "data", "customer_docs")
INTERNAL_DOCS_DIR = os.path.join(BASE_DIR, "data", "internal_docs")
INDEX_DIR = os.path.join(BASE_DIR, "index")

# --- Demo auth for internal (employee) mode ---
# NOTE: this is a placeholder for the prototype/demo only.
# In a real deployment, replace with proper SSO / role-based auth.
INTERNAL_MODE_PASSWORD = os.getenv("INTERNAL_MODE_PASSWORD", "employee123")
