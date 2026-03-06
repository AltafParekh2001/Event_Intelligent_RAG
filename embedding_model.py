from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

# Loaded once and reused throughout the application
_model = None


def get_model() -> SentenceTransformer:
    """Return the singleton embedding model (lazy-loaded)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_embedding_dimension() -> int:
    """Return the dimension of the embedding vectors."""
    return get_model().get_sentence_embedding_dimension()


def embed_texts(texts: list):
    """Encode a list of strings into embeddings (numpy array)."""
    return get_model().encode(texts, show_progress_bar=False)


# Convenience alias used by retriever.py
model = get_model()