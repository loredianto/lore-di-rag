"""Caricamento dei documenti sorgente."""

from .loaders import LoadedDocument, discover_input_files, load_documents

__all__ = ["LoadedDocument", "discover_input_files", "load_documents"]
