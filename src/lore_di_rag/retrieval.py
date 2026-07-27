"""Ricerca densa sui piccoli indici NumPy prodotti dalla pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .chunking import Chunk
from .embeddings import Embedder
from .storage import StoredIndex, load_index


@dataclass(frozen=True)
class SearchResult:
    """Chunk recuperato, con posizione e score di similarità."""

    rank: int
    score: float
    chunk: Chunk


def rank_chunks(
    index: StoredIndex,
    query_embedding: NDArray[np.floating],
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    """Ordina i chunk per prodotto scalare con un singolo embedding di query."""

    if top_k <= 0:
        raise ValueError("top_k deve essere maggiore di zero")

    embeddings = np.asarray(index.embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(index.chunks):
        raise ValueError("La matrice dell'indice non è allineata ai chunk")
    if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
        raise ValueError("Non è possibile cercare in un indice vuoto")
    if not np.isfinite(embeddings).all():
        raise ValueError("L'indice contiene valori non finiti")

    query_vector = np.asarray(query_embedding, dtype=np.float32)
    if query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise ValueError("L'embedding della query deve avere forma (1, dimensioni)")
    if not np.isfinite(query_vector).all():
        raise ValueError("L'embedding della query contiene valori non finiti")
    if query_vector.shape[1] != embeddings.shape[1]:
        raise ValueError(
            "Dimensione query non compatibile con l'indice: "
            f"{query_vector.shape[1]} != {embeddings.shape[1]}"
        )

    scores = embeddings @ query_vector[0]
    ordered_indices = np.argsort(-scores, kind="stable")[: min(top_k, len(scores))]
    return [
        SearchResult(
            rank=rank,
            score=float(scores[chunk_index]),
            chunk=index.chunks[int(chunk_index)],
        )
        for rank, chunk_index in enumerate(ordered_indices, start=1)
    ]


def search_stored_index(
    index: StoredIndex,
    query: str,
    embedder: Embedder,
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    """Codifica una query e cerca in un indice già caricato."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("La query non può essere vuota")

    indexed_model = index.manifest.get("embedding", {}).get("model")
    if indexed_model != embedder.model_name:
        raise ValueError(
            "Il modello della query non coincide con quello dell'indice: "
            f"{embedder.model_name!r} != {indexed_model!r}"
        )

    return rank_chunks(
        index,
        embedder.encode([normalized_query]),
        top_k=top_k,
    )


def search_index(
    index_dir: Path,
    query: str,
    embedder: Embedder,
    *,
    top_k: int = 5,
) -> list[SearchResult]:
    """Carica un indice locale e restituisce i chunk più rilevanti."""

    return search_stored_index(
        load_index(index_dir),
        query,
        embedder,
        top_k=top_k,
    )
