"""
CLI: build the RAG knowledge base end-to-end.

Runs the full pipeline against knowledge_base/pdfs/:
    chunk PDFs -> embed chunks -> upsert into ChromaDB

Idempotent by construction (deterministic IDs + upsert), so re-running
after adding a new PDF only writes the new chunks; existing chunks are
overwritten in place.

Usage:
    python scripts/build_kb.py
"""

import argparse
import logging
import sys
from pathlib import Path

from aml_agent.rag.chunker import chunk_directory
from aml_agent.rag.embedder import embed_chunks
from aml_agent.rag.store import DEFAULT_STORE_DIR, store_chunks


DEFAULT_PDF_DIR = Path("knowledge_base/pdfs")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AML RAG knowledge base.")
    parser.add_argument(
        "--pdf-dir", type=Path, default=DEFAULT_PDF_DIR,
        help="Directory of source PDFs (default: knowledge_base/pdfs).",
    )
    parser.add_argument(
        "--store-dir", type=Path, default=DEFAULT_STORE_DIR,
        help="ChromaDB persistence directory (default: knowledge_base/chroma).",
    )
    args = parser.parse_args()

    _configure_logging()
    logger = logging.getLogger("aml_agent.rag.build")

    if not args.pdf_dir.exists():
        logger.error("PDF directory not found: %s", args.pdf_dir)
        sys.exit(1)

    logger.info("Chunking PDFs from %s", args.pdf_dir)
    chunks = chunk_directory(args.pdf_dir)

    logger.info("Embedding and upserting into ChromaDB at %s", args.store_dir)
    # embed_chunks is a generator, store_chunks consumes it — the whole
    # pipeline streams end-to-end, so peak memory stays bounded regardless
    # of corpus size.
    total = store_chunks(embed_chunks(chunks), store_dir=args.store_dir)

    logger.info("Knowledge base built: %d chunks written to collection", total)
    print(f"Wrote {total} chunks to {args.store_dir}")


if __name__ == "__main__":
    main()