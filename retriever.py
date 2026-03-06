import os
import logging
from datetime import datetime
from embedding_model import get_model
from vector_store import search_vectors

logger = logging.getLogger(__name__)

RETRIEVED_LOG = "retrieved_chunks.log"


def retrieve_documents(question):
    """
    Embed question, return top matching chunks from ChromaDB,
    print them to terminal AND save them to retrieved_chunks.log.
    """
    query_embedding = get_model().encode(question)
    results = search_vectors(query_embedding)
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # ── Print to terminal ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"📦 RETRIEVED CHUNKS ({len(documents)} found for your question)")
    print(f"{'='*60}")
    for i, (doc, dist) in enumerate(zip(documents, distances)):
        similarity = round((1 - dist) * 100, 1)
        print(f"\n--- Chunk {i+1} (Similarity: {similarity}%) ---")
        print(doc[:800])   # show first 800 chars of each chunk
        if len(doc) > 800:
            print(f"  ... [{len(doc) - 800} more chars not shown]")
    print(f"\n{'='*60}\n")

    # ── Save to file ──────────────────────────────────────────────
    with open(RETRIEVED_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'#'*60}\n")
        f.write(f"TIME     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"QUESTION : {question}\n")
        f.write(f"CHUNKS   : {len(documents)} retrieved\n")
        f.write(f"{'#'*60}\n")
        for i, (doc, dist) in enumerate(zip(documents, distances)):
            similarity = round((1 - dist) * 100, 1)
            f.write(f"\n--- Chunk {i+1} (Similarity: {similarity}%) ---\n")
            f.write(doc + "\n")
        f.write(f"\n{'='*60}\n")

    logger.info("Retrieved %d chunks → saved to %s", len(documents), RETRIEVED_LOG)
    return documents