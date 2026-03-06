"""
test_validation.py
Validates all modules of the Event RAG System.
Run with:  python test_validation.py
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── Individual test functions ────────────────────────────────────

def test_requirements() -> bool:
    logger.info("\nChecking installed packages...")
    required = [
        "pandas", "sqlalchemy", "chromadb",
        "sentence_transformers", "nltk", "requests", "dotenv",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            logger.info("  ✓ %s", pkg)
        except ImportError:
            missing.append(pkg)
            logger.error("  ✗ %s NOT installed", pkg)

    if missing:
        logger.error("Run: pip install -r requirements.txt")
        return False
    return True


def test_imports() -> bool:
    logger.info("\nImporting project modules...")
    modules = [
        "config", "ingest_data", "feature_engineering",
        "text_chunking", "embedding_model", "vector_store",
        "retriever", "rag_pipeline",
    ]
    for mod in modules:
        try:
            __import__(mod)
            logger.info("  ✓ %s", mod)
        except Exception as exc:
            logger.error("  ✗ %s – %s", mod, exc)
            return False
    return True


def test_config() -> bool:
    logger.info("\nValidating configuration...")
    try:
        from config import (
            DB_PATH, CHROMA_DB_PATH, EMBEDDING_MODEL_NAME,
            CHUNK_SIZE, MAX_NEW_TOKENS, NUM_RETRIEVED_DOCS,
        )
        assert DB_PATH
        assert CHROMA_DB_PATH
        assert EMBEDDING_MODEL_NAME
        assert CHUNK_SIZE > 0
        assert MAX_NEW_TOKENS > 0
        assert NUM_RETRIEVED_DOCS > 0
        logger.info("  ✓ All config values present and valid")
        return True
    except Exception as exc:
        logger.error("  ✗ %s", exc)
        return False


def test_text_chunking() -> bool:
    logger.info("\nTesting text chunking...")
    try:
        from text_chunking import chunk_text, get_chunk_statistics
        sample = "This is a test sentence. " * 50
        chunks = chunk_text(sample)
        assert chunks and all(isinstance(c, str) for c in chunks)
        stats = get_chunk_statistics(chunks)
        assert stats["total_chunks"] > 0
        logger.info("  ✓ %d chunks created", stats["total_chunks"])
        return True
    except Exception as exc:
        logger.error("  ✗ %s", exc)
        return False


def test_embedding_model() -> bool:
    logger.info("\nTesting embedding model...")
    try:
        from embedding_model import get_model, get_embedding_dimension
        model = get_model()
        assert model is not None
        dim = get_embedding_dimension()
        assert dim > 0
        logger.info("  ✓ Model ready (dimension: %d)", dim)
        return True
    except Exception as exc:
        logger.error("  ✗ %s", exc)
        return False


def test_vector_store() -> bool:
    logger.info("\nTesting vector store...")
    try:
        from vector_store import get_client, get_collection, get_collection_stats
        assert get_client() is not None
        assert get_collection() is not None
        stats = get_collection_stats()
        assert "total_documents" in stats
        logger.info("  ✓ Vector store ready (%d docs)", stats["total_documents"])
        return True
    except Exception as exc:
        logger.error("  ✗ %s", exc)
        return False


def test_rag_pipeline() -> bool:
    logger.info("\nTesting LLM connection (HF Inference API)...")
    try:
        from rag_pipeline import test_llm_connection
        ok = test_llm_connection()
        if ok:
            logger.info("  ✓ HF API reachable")
        else:
            logger.warning(
                "  ⚠ HF API did not respond – check HF_TOKEN in .env "
                "or model may still be loading"
            )
        return True   # connection failure is non-fatal
    except Exception as exc:
        logger.error("  ✗ %s", exc)
        return False


# ─── Runner ───────────────────────────────────────────────────────

def main() -> int:
    logger.info("=" * 60)
    logger.info("EVENT RAG SYSTEM – VALIDATION TESTS")
    logger.info("=" * 60)

    tests = [
        ("Requirements",    test_requirements),
        ("Imports",         test_imports),
        ("Configuration",   test_config),
        ("Text Chunking",   test_text_chunking),
        ("Embedding Model", test_embedding_model),
        ("Vector Store",    test_vector_store),
        ("RAG Pipeline",    test_rag_pipeline),
    ]

    results = []
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except Exception as exc:
            logger.error("Test %s crashed: %s", name, exc)
            results.append((name, False))

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        logger.info("  %s %s", "✓ PASS" if result else "✗ FAIL", name)

    logger.info("\n%d / %d tests passed", passed, len(results))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
