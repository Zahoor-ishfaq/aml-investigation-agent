"""
ChromaDB storage for the RAG knowledge base.

Persists (chunk, vector) pairs to a local, file-backed ChromaDB collection.
Same-process client (no server) matches the local-dev infra pattern used
across the project.

Design notes:
- Deterministic IDs ({source}::{chunk_index}) make ingestion idempotent:
  re-running against the same PDFs overwrites existing rows instead of
  duplicating them. Critical because Chroma's `add()` errors on duplicate
  IDs; `upsert()` doesn't.
- Cosine similarity index (space="cosine") pairs with the normalize=True
  embedding step in embedder.py — the two together let Chroma use its
  fast inner-product path while still computing cosine semantics.
"""

from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings

from aml_agent.rag.chunker import Chunk


COLLECTION_NAME = "aml_typologies"
DEFAULT_STORE_DIR = Path("knowledge_base/chroma")


def get_collection(store_dir: Path = DEFAULT_STORE_DIR):
    """
    Return (or create) the aml_typologies collection.

    PersistentClient writes to disk under store_dir; the folder is
    created lazily on first use. anonymized_telemetry=False disables
    Chroma's optional usage pings — off by default in modern versions
    but explicit here so behavior doesn't change if a future upgrade
    flips the default back.

    cosine metric matches our normalized embeddings; changing this
    without also changing embedder.py's normalize_embeddings flag would
    silently degrade retrieval quality.
    """
    store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(store_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(chunk: Chunk) -> str:
    """Deterministic ID: same source PDF + same chunk position = same ID.
    Enables idempotent re-ingest via upsert."""
    return f"{chunk.source}::{chunk.chunk_index}"


def store_chunks(
    embedded: Iterable[tuple[Chunk, list[float]]],
    batch_size: int = 100,
    store_dir: Path = DEFAULT_STORE_DIR,
) -> int:
    """
    Upsert (chunk, vector) pairs into the collection in batches.

    Batched because Chroma's per-call overhead dominates on single-row
    inserts — 100 rows/batch is well under Chroma's internal max and
    keeps memory bounded regardless of corpus size. Returns total rows
    written for the caller's progress reporting.
    """
    collection = get_collection(store_dir)

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []
    written = 0

    def flush() -> None:
        nonlocal written
        if not ids:
            return
        # upsert (not add) so re-ingesting the same corpus is safe —
        # add() would raise on duplicate IDs.
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        written += len(ids)
        ids.clear()
        documents.clear()
        embeddings.clear()
        metadatas.clear()

    for chunk, vector in embedded:
        ids.append(_chunk_id(chunk))
        documents.append(chunk.text)
        embeddings.append(vector)
        metadatas.append({
            "source": chunk.source,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
        })
        if len(ids) >= batch_size:
            flush()

    flush()
    return written