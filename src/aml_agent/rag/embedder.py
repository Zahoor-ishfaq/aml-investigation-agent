"""
Embedding pipeline for the RAG knowledge base.

Takes Chunk objects (from chunker.py) and produces (chunk, vector) pairs
using sentence-transformers/all-MiniLM-L6-v2. Storage is 5.4's concern;
this module only produces vectors.

Model choice: all-MiniLM-L6-v2, 384-dim. Small, fast, runs on CPU. Standard
baseline across the RAG ecosystem; sufficient for a corpus of hundreds
of documents. Reference: Reimers & Gurevych, "Sentence-BERT",
arXiv:1908.10084. Larger models (all-mpnet-base-v2, 768-dim) exist but
add cost without proportional accuracy gain at our corpus size.
"""

from functools import lru_cache
from typing import Iterable, Iterator

from sentence_transformers import SentenceTransformer

from aml_agent.rag.chunker import Chunk


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Batch size for encoding. 32 is the sentence-transformers default and
# balances throughput vs memory on CPU. Raising helps if GPU is available;
# not needed for our corpus size.
BATCH_SIZE = 32


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Lazy-loaded singleton. First call downloads the model to
    ~/.cache/huggingface/ (~80MB, one-time). lru_cache guarantees the
    (heavy) constructor runs only once per process — importing this
    module has no side effect until embed_* is called.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Encode a batch of texts to 384-dim vectors.

    convert_to_numpy=False + list output keeps ChromaDB happy (it expects
    plain Python lists, not numpy arrays, in its add() calls) and avoids
    a downstream conversion. normalize_embeddings=True means cosine
    similarity == dot product, which lets ChromaDB use its faster
    inner-product index without sacrificing correctness.
    """
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=False,
    )
    return [list(map(float, v)) for v in vectors]


def embed_chunks(chunks: Iterable[Chunk]) -> Iterator[tuple[Chunk, list[float]]]:
    """
    Embed a stream of chunks in batches, yield (chunk, vector) pairs.

    Buffered batching (not one-shot list materialisation) so the caller
    can pipe a very large corpus through without holding all vectors in
    memory at once — matters if the KB grows to thousands of PDFs later.
    """
    buffer: list[Chunk] = []
    for chunk in chunks:
        buffer.append(chunk)
        if len(buffer) >= BATCH_SIZE:
            yield from _flush(buffer)
            buffer = []
    if buffer:
        yield from _flush(buffer)


def _flush(buffer: list[Chunk]) -> Iterator[tuple[Chunk, list[float]]]:
    """Encode the current buffer and yield paired results in order."""
    vectors = embed_texts([c.text for c in buffer])
    for chunk, vec in zip(buffer, vectors):
        yield chunk, vec