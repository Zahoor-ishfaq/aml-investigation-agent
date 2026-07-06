"""
PDF chunking for the RAG knowledge base.

Reads PDFs from knowledge_base/pdfs/, extracts text page-by-page,
splits into overlapping chunks, returns records with per-page metadata
so agent citations can point at specific pages of specific documents
(FATF R20 requires per-source traceability of investigative conclusions).

No embedding or DB writes here — that's substep 5.3 (embeddings) and
5.4 (ChromaDB ingestion). This module produces the intermediate stream
those steps consume.

Reference: pypdf docs, https://pypdf.readthedocs.io/
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader


# Chunk sizing follows the empirical defaults across the RAG ecosystem
# (LangChain RecursiveCharacterTextSplitter, LlamaIndex SimpleNodeParser).
# 2000 chars ≈ 500 tokens fits well inside all-MiniLM-L6-v2's 512-token
# window with headroom; 200-char overlap preserves context at boundaries
# where a sentence gets split across chunks.
CHUNK_CHARS = 2000
OVERLAP_CHARS = 200
MIN_CHUNK_CHARS = 100  # discards page headers/footers, blank-page artifacts


@dataclass(frozen=True)
class Chunk:
    """One chunk of extracted text with source attribution."""
    source: str        # filename, e.g. "fatf_recommendations_2012.pdf"
    page: int          # 1-indexed page number (matches how humans cite)
    chunk_index: int   # position within the source, 0-indexed
    text: str


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a page's text into overlapping chunks using a paragraph -> sentence
    -> char fallback hierarchy. Preserves semantic boundaries where possible:
    a chunk ends at a paragraph break if one is available, at a sentence
    otherwise, at a hard char cut only as last resort. Overlap is measured
    in chars (not tokens) — good enough approximation for MiniLM's
    subword tokenizer without pulling in tiktoken just for boundary math.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer breaking at a paragraph boundary within the last 20% of
        # the chunk window — falls back to sentence, then hard cut.
        if end < len(text):
            search_from = start + int(chunk_size * 0.8)
            para = text.rfind("\n\n", search_from, end)
            if para != -1:
                end = para
            else:
                sent = text.rfind(". ", search_from, end)
                if sent != -1:
                    end = sent + 1  # keep the period with the preceding chunk

        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

        # Advance with overlap. If end < start + overlap we'd loop forever
        # on tiny remaining tails, so guard that.
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def chunk_pdf(pdf_path: Path) -> Iterator[Chunk]:
    """
    Extract text page-by-page and yield Chunk records.

    Per-page extraction (not whole-doc-then-chunk) means each chunk carries
    a precise page number for citation. Rare case: a paragraph spans two
    pages — we accept that boundary rather than merging pages, because
    faithful page attribution is worth more to a compliance officer than
    perfect paragraph coherence.
    """
    reader = PdfReader(str(pdf_path))
    source = pdf_path.name
    global_idx = 0

    for page_num, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        for chunk_text in _split_text(raw, CHUNK_CHARS, OVERLAP_CHARS):
            yield Chunk(
                source=source,
                page=page_num,
                chunk_index=global_idx,
                text=chunk_text,
            )
            global_idx += 1


def chunk_directory(pdf_dir: Path) -> Iterator[Chunk]:
    """Yield chunks across every PDF in a directory. Sorted for
    deterministic ordering — reruns produce identical chunk_index
    sequences, which matters when comparing embedding runs."""
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        yield from chunk_pdf(pdf_path)