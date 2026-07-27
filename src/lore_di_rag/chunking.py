"""Suddivisione deterministica dei documenti prima dell'embedding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .ingestion import LoadedDocument


@dataclass(frozen=True)
class Chunk:
    """Unità testuale indicizzata, collegata al documento sorgente."""

    text: str
    source_path: Path
    page_number: int | None
    chunk_index: int


def _preferred_end(text: str, start: int, hard_end: int) -> int:
    """Preferisce un confine di paragrafo, poi uno spazio, evitando pezzi minuscoli."""

    minimum_end = start + max(1, (hard_end - start) // 2)
    paragraph_end = text.rfind("\n\n", minimum_end, hard_end)
    if paragraph_end >= minimum_end:
        return paragraph_end

    whitespace_end = -1
    for match in re.finditer(r"\s+", text[minimum_end:hard_end]):
        whitespace_end = minimum_end + match.start()
    return whitespace_end if whitespace_end >= minimum_end else hard_end


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Crea finestre paragraph-aware espresse in caratteri.

    I confini di paragrafo sono preferiti quando possibile. I paragrafi troppo
    lunghi usano un taglio su whitespace e, come ultimo fallback, un taglio netto.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size deve essere maggiore di zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap deve rispettare 0 <= overlap < chunk_size")

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(normalized)

    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = (
            _preferred_end(normalized, start, hard_end)
            if hard_end < text_length
            else hard_end
        )
        if end <= start:
            end = hard_end

        chunk = normalized[start:end].strip()
        if chunk and (not chunks or chunk != chunks[-1]):
            chunks.append(chunk)
        if end >= text_length:
            break

        next_start = max(0, end - overlap)
        while next_start < end and normalized[next_start].isspace():
            next_start += 1
        start = next_start if next_start > start else end

    return chunks


def chunk_documents(
    documents: list[LoadedDocument],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[Chunk]:
    """Suddivide ogni documento mantenendo source, pagina e indice del chunk."""

    chunks: list[Chunk] = []
    for document in documents:
        for chunk_index, text in enumerate(
            chunk_text(document.text, chunk_size=chunk_size, overlap=overlap)
        ):
            chunks.append(
                Chunk(
                    text=text,
                    source_path=document.source_path,
                    page_number=document.page_number,
                    chunk_index=chunk_index,
                )
            )
    return chunks
