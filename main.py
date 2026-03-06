"""
main.py – Entry point for the Event RAG System
Uses the free Hugging Face Inference API (no OpenAI, no Ollama).
"""

import logging
import sys
from config import LOG_LEVEL, LOG_FILE

# ─── Logging setup (do this before importing project modules) ─────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)

# ─── Project imports ──────────────────────────────────────────────
from ingest_data import ingest_csv
from feature_engineering import create_event_text
from text_chunking import chunk_text, get_chunk_statistics
from embedding_model import embed_texts
from vector_store import store_vectors
from retriever import retrieve_documents
from rag_pipeline import generate_answer


# ─── Core functions ───────────────────────────────────────────────

def build_rag_system(csv_path: str) -> None:
    """Ingest CSV data and build the vector store from scratch."""
    logger.info("Building RAG system from: %s", csv_path)

    df = ingest_csv(csv_path)
    df = create_event_text(df)

    chunks: list = []
    for text in df["event_text"]:
        chunks.extend(chunk_text(text))

    stats = get_chunk_statistics(chunks)
    logger.info(
        "Chunking complete – %d chunks (avg %d chars)",
        stats["total_chunks"], stats["avg_length"],
    )

    embeddings = embed_texts(chunks)
    store_vectors(chunks, embeddings)

    logger.info("RAG system built successfully (%d vectors indexed)", len(chunks))
    print("\n✓ RAG system ready.\n")


def ask_question(question: str) -> None:
    """Retrieve relevant context and generate an answer."""
    docs = retrieve_documents(question)
    context = "\n\n".join(docs)
    answer = generate_answer(context, question)
    print(f"\n{'─'*60}\nAnswer:\n{answer}\n{'─'*60}\n")


# ─── CLI entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    csv_file = "V_EVENT_DETAILS_202512311554.csv"

    build_rag_system(csv_file)

    print("Type your question below (Ctrl+C to exit).\n")
    try:
        while True:
            question = input("Ask a question: ").strip()
            if question:
                ask_question(question)
    except KeyboardInterrupt:
        print("\nGoodbye!")