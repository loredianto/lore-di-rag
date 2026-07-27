"""Persistenza validata di embedding, chunk e manifest dell'indice."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .chunking import Chunk


EMBEDDINGS_FILENAME = "embeddings.npy"
CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StoredIndex:
    embeddings: NDArray[np.float32]
    chunks: list[Chunk]
    manifest: dict[str, Any]


def _validate_embeddings(
    embeddings: NDArray[np.floating[Any]],
    expected_rows: int,
) -> NDArray[np.float32]:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("embeddings deve essere una matrice 2D")
    if array.shape[0] != expected_rows:
        raise ValueError(
            f"Numero di chunk ({expected_rows}) diverso dalle righe embedding "
            f"({array.shape[0]})"
        )
    if expected_rows == 0 or array.shape[1] == 0:
        raise ValueError("Non è possibile salvare un indice vuoto")
    if not np.isfinite(array).all():
        raise ValueError("Gli embedding contengono valori non finiti")
    return array


def _chunk_record(chunk: Chunk) -> dict[str, Any]:
    return {
        "text": chunk.text,
        "source": str(chunk.source_path),
        "page": chunk.page_number,
        "chunk_index": chunk.chunk_index,
    }


def _write_temp_bytes(directory: Path, suffix: str, content: bytes) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=directory, prefix=".tmp-", suffix=suffix, delete=False
    ) as temporary_file:
        temporary_file.write(content)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
        return Path(temporary_file.name)


def save_index(
    index_dir: Path,
    embeddings: NDArray[np.floating[Any]],
    chunks: Sequence[Chunk],
    *,
    model_name: str,
    chunk_size: int,
    overlap: int,
    embedding_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Salva i tre artefatti dell'indice con file temporanei e rename atomici."""

    chunks_list = list(chunks)
    array = _validate_embeddings(embeddings, len(chunks_list))
    index_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "documents": len({str(chunk.source_path) for chunk in chunks_list}),
        "chunks": len(chunks_list),
        "embedding": {
            **dict(embedding_options or {}),
            "model": model_name,
            "dimensions": int(array.shape[1]),
        },
        "chunking": {
            "strategy": "paragraph_aware",
            "unit": "characters",
            "size": chunk_size,
            "overlap": overlap,
        },
        "files": {
            "embeddings": EMBEDDINGS_FILENAME,
            "chunks": CHUNKS_FILENAME,
        },
    }

    chunk_bytes = "".join(
        json.dumps(_chunk_record(chunk), ensure_ascii=False) + "\n"
        for chunk in chunks_list
    ).encode("utf-8")
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    temporary_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=index_dir, prefix=".tmp-", suffix=".npy", delete=False
        ) as temporary_embeddings:
            np.save(temporary_embeddings, array, allow_pickle=False)
            temporary_embeddings.flush()
            os.fsync(temporary_embeddings.fileno())
            embeddings_temp_path = Path(temporary_embeddings.name)
        temporary_paths.append(embeddings_temp_path)

        chunks_temp_path = _write_temp_bytes(index_dir, ".jsonl", chunk_bytes)
        temporary_paths.append(chunks_temp_path)
        manifest_temp_path = _write_temp_bytes(index_dir, ".json", manifest_bytes)
        temporary_paths.append(manifest_temp_path)

        os.replace(embeddings_temp_path, index_dir / EMBEDDINGS_FILENAME)
        os.replace(chunks_temp_path, index_dir / CHUNKS_FILENAME)
        os.replace(manifest_temp_path, index_dir / MANIFEST_FILENAME)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)

    return manifest


def load_index(index_dir: Path) -> StoredIndex:
    """Carica un indice senza pickle e ne verifica consistenza e versione."""

    manifest_path = index_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Versione del formato indice non supportata")

    embeddings = np.asarray(
        np.load(index_dir / EMBEDDINGS_FILENAME, allow_pickle=False),
        dtype=np.float32,
    )
    chunks: list[Chunk] = []
    with (index_dir / CHUNKS_FILENAME).open(encoding="utf-8") as chunks_file:
        for line_number, line in enumerate(chunks_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                chunks.append(
                    Chunk(
                        text=str(record["text"]),
                        source_path=Path(record["source"]),
                        page_number=record.get("page"),
                        chunk_index=int(record["chunk_index"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Record chunk non valido alla riga {line_number}"
                ) from error

    embeddings = _validate_embeddings(embeddings, len(chunks))
    expected_chunks = int(manifest.get("chunks", -1))
    expected_dimensions = int(manifest.get("embedding", {}).get("dimensions", -1))
    if expected_chunks != len(chunks) or expected_dimensions != embeddings.shape[1]:
        raise ValueError("Manifest non coerente con i file dell'indice")

    return StoredIndex(embeddings=embeddings, chunks=chunks, manifest=manifest)
