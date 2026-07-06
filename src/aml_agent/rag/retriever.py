"""
RAG retrieval.

Given a natural-language query, return the top-k most similar chunks
from the AML typology knowledge base. Called by the tool layer (Phase 4
will get a new tool wrapping this) and eventually by the agent (Phase 7).

Query-time embedding MUST use the same model as ingestion (embedder.py);
different models produce different vector spaces and cosine similarity
across spaces is meaningless. The shared _get_model() singleton in
embedder.py enforces this — both paths go through it.
"""

from pathlib import Path
from typing import Optional

from aml_agent.rag.embedder import embed_texts
from aml_agent.rag.store import DEFAULT_STORE_DIR, get_collection


def retrieve(
    query: str,
    k: int = 3,
    source_filter: Optional[str] = None,
    store_dir: Path = DEFAULT_STORE_DIR,
) -> list[dict]:
    """
    Return the top-k chunks most similar to `query`.

    Each result: {text, source, page, chunk_index, similarity_score}.

    similarity_score is in [0, 1], higher is closer. Chroma returns a
    "distance" in cosine space; we convert to similarity as (1 - distance)
    so callers reason about a monotonically-increasing quality signal
    without having to know Chroma's metric convention.

    source_filter narrows to a single source PDF's chunks — useful when
    the agent already knows it wants FATF-specific vs Wolfsberg-specific
    guidance. Passed through as Chroma's `where` metadata filter.
    """
    collection = get_collection(store_dir)

    query_vec = embed_texts([query])[0]

    where = {"source": source_filter} if source_filter else None

    result = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        where=where,
        # Ask for exactly the fields we return — avoids paying to
        # deserialize embeddings we don't need in the response.
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns column-oriented results (one list per query); we
    # requested one query so index [0] into each list to get per-hit rows.
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    return [
        {
            "text": doc,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
            "similarity_score": 1.0 - float(dist),
        }
        for doc, meta, dist in zip(docs, metas, dists)
    ]