from __future__ import annotations

import logging
import chromadb
from config import CHROMA_DB_PATH, NUM_RETRIEVED_DOCS
from typing import Optional

logger = logging.getLogger(__name__)

_client: Optional[chromadb.Client] = None
_collection = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name="event_collection",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def store_vectors(texts: list, embeddings) -> None:
    """Store text chunks and their embeddings in ChromaDB in safe batches."""
    collection = get_collection()
    ids = [f"id_{i}" for i in range(len(texts))]
    embeddings_list = [e.tolist() for e in embeddings]

    BATCH_SIZE = 500  # safely under ChromaDB's hard limit of 5461
    total = len(texts)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.upsert(
            documents=texts[start:end],
            embeddings=embeddings_list[start:end],
            ids=ids[start:end],
        )
        logger.info("Stored batch %d-%d of %d vectors", start + 1, end, total)

    logger.info("All %d vectors stored in ChromaDB (%s)", total, CHROMA_DB_PATH)


def search_vectors(query_embedding, n_results: int = NUM_RETRIEVED_DOCS) -> dict:
    """Return the n_results most similar documents for query_embedding."""
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
    )


def get_collection_stats() -> dict:
    try:
        return {"total_documents": get_collection().count()}
    except Exception as exc:
        logger.warning("Could not fetch collection stats: %s", exc)
        return {"total_documents": 0}