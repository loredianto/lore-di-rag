"""Wrapper per gli embedding densi BGE-M3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import DEFAULT_BATCH_SIZE, DEFAULT_MAX_LENGTH, DEFAULT_MODEL_NAME


class Embedder(Protocol):
    """Contratto minimo usato da pipeline e retrieval."""

    model_name: str

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Restituisce una matrice 2D float32, una riga per testo."""


@dataclass
class BGEM3Embedder:
    """Caricamento lazy di BGE-M3, così import e test non scaricano il modello."""

    model_name: str = DEFAULT_MODEL_NAME
    use_fp16: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    _model: Any = field(default=None, init=False, repr=False)

    def _get_model(self) -> Any:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
        return self._model

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        values = list(texts)
        if not values:
            raise ValueError("Serve almeno un testo da embeddare")

        output = self._get_model().encode(
            values,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        embeddings = np.asarray(output["dense_vecs"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(values):
            raise ValueError("Il modello ha restituito una matrice di forma inattesa")
        if not np.isfinite(embeddings).all():
            raise ValueError("Gli embedding contengono valori non finiti")
        return embeddings
