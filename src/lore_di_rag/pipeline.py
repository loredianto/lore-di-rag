"""Orchestrazione della pipeline offline che costruisce l'indice locale."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .chunking import chunk_documents
from .config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from .embeddings import Embedder
from .ingestion import load_documents
from .storage import save_index


@dataclass(frozen=True)
class IndexBuildResult:
    """Riepilogo dell'indice appena costruito."""

    index_dir: Path
    document_count: int
    chunk_count: int
    embedding_dimensions: int
    manifest: dict[str, Any]


def build_index(
    input_dir: Path,
    index_dir: Path,
    embedder: Embedder,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_options: Mapping[str, Any] | None = None,
) -> IndexBuildResult:
    """Esegue load → chunk → embed → salvataggio dell'indice.

    Il chunking precede intenzionalmente l'embedding: ogni riga di
    ``embeddings.npy`` deve rappresentare esattamente un record di
    ``chunks.jsonl``.
    """

    documents = load_documents(input_dir)
    if not documents:
        raise ValueError(f"Nessun documento leggibile trovato in: {input_dir}")

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        raise ValueError("I documenti non hanno prodotto alcun chunk")

    embeddings = embedder.encode([chunk.text for chunk in chunks])
    manifest = save_index(
        index_dir,
        embeddings,
        chunks,
        model_name=embedder.model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        embedding_options=embedding_options,
    )

    return IndexBuildResult(
        index_dir=index_dir.expanduser().resolve(),
        document_count=int(manifest["documents"]),
        chunk_count=int(manifest["chunks"]),
        embedding_dimensions=int(manifest["embedding"]["dimensions"]),
        manifest=manifest,
    )
