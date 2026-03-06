import logging
import nltk
from config import CHUNK_SIZE

logger = logging.getLogger(__name__)

# Download punkt tokenizer data on first use
for _resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.tokenize import sent_tokenize  # noqa: E402  (must come after download)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list:
    """
    Split *text* into chunks that fit within *chunk_size* characters,
    keeping sentence boundaries intact.
    """
    sentences = sent_tokenize(text)
    chunks: list = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def get_chunk_statistics(chunks: list) -> dict:
    """Return basic statistics about a list of text chunks."""
    if not chunks:
        return {"total_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}

    lengths = [len(c) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_length": sum(lengths) // len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }