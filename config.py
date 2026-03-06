import os
from dotenv import load_dotenv

load_dotenv()

# ─── Groq (free, no credit card required) ────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"        # fast, free
# GROQ_MODEL = "llama-3.3-70b-versatile"     # smarter, also free

# ─── Embedding model (runs locally, 100% free) ───────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ─── Storage ─────────────────────────────────────────────────────
DB_PATH        = os.getenv("DB_PATH",        "events.db")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# ─── RAG tuning ──────────────────────────────────────────────────
CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE",         "1500"))
MAX_NEW_TOKENS     = int(os.getenv("MAX_NEW_TOKENS",     "512"))
# Keep retrieved docs low to avoid hitting the 6000 TPM free tier limit
NUM_RETRIEVED_DOCS = int(os.getenv("NUM_RETRIEVED_DOCS", "3"))
TEMPERATURE        = float(os.getenv("TEMPERATURE",      "0.2"))

# ─── Logging ─────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE  = os.getenv("LOG_FILE",  "rag_system.log")