"""Loader per i formati di documento accettati dalla pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TEXT_EXTENSIONS = frozenset({".md", ".rst", ".txt"})
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf"}


@dataclass(frozen=True)
class LoadedDocument:
    """Testo estratto da un file, con le informazioni sulla sua origine."""

    text: str
    source_path: Path
    page_number: int | None = None


def discover_input_files(input_dir: Path) -> list[Path]:
    """Restituisce in ordine stabile tutti i file supportati nella directory."""

    if not input_dir.exists():
        raise FileNotFoundError(f"Directory di input non trovata: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Il percorso di input non è una directory: {input_dir}")

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _load_text_file(path: Path) -> list[LoadedDocument]:
    text = path.read_text(encoding="utf-8").strip()
    return [LoadedDocument(text=text, source_path=path)] if text else []


def _load_pdf(path: Path) -> list[LoadedDocument]:
    from pypdf import PdfReader

    documents = []
    for page_number, page in enumerate(PdfReader(path).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                LoadedDocument(
                    text=text,
                    source_path=path,
                    page_number=page_number,
                )
            )
    return documents


def load_documents(input_dir: Path) -> list[LoadedDocument]:
    """Carica PDF e file testuali presenti nella directory di input."""

    documents = []
    for path in discover_input_files(input_dir):
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path))
        else:
            documents.extend(_load_text_file(path))
    return documents
