"""Interfaccia a riga di comando per indicizzazione e ricerca."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    ProjectPaths,
)
from .embeddings import BGEM3Embedder
from .pipeline import build_index
from .retrieval import search_stored_index
from .storage import load_index


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("il valore deve essere maggiore di zero")
    return parsed


def create_parser(project_root: Path | None = None) -> argparse.ArgumentParser:
    """Crea il parser; ``project_root`` rende testabili i percorsi di default."""

    paths = ProjectPaths.from_root(project_root or Path.cwd())
    parser = argparse.ArgumentParser(
        prog="lore-di-rag",
        description="Indicizza documenti locali e cerca i chunk più pertinenti.",
    )
    commands = parser.add_subparsers(dest="command")

    index_parser = commands.add_parser(
        "index",
        help="Crea un indice da PDF e file testuali.",
    )
    index_parser.add_argument("--input-dir", type=Path, default=paths.input_dir)
    index_parser.add_argument("--index-dir", type=Path, default=paths.index_dir)
    index_parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    index_parser.add_argument(
        "--chunk-size",
        type=_positive_integer,
        default=DEFAULT_CHUNK_SIZE,
    )
    index_parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
    )
    index_parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=DEFAULT_BATCH_SIZE,
    )
    index_parser.add_argument(
        "--max-length",
        type=_positive_integer,
        default=DEFAULT_MAX_LENGTH,
    )
    index_parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Abilita FP16 se supportato dall'hardware.",
    )

    search_parser = commands.add_parser(
        "search",
        help="Cerca semanticamente in un indice esistente.",
    )
    search_parser.add_argument("query", help="Domanda o testo da cercare.")
    search_parser.add_argument("--index-dir", type=Path, default=paths.index_dir)
    search_parser.add_argument(
        "--top-k",
        type=_positive_integer,
        default=5,
        help="Numero massimo di chunk da mostrare.",
    )
    search_parser.add_argument(
        "--model",
        default=None,
        help="Modello della query; per default usa quello registrato nell'indice.",
    )
    search_parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=DEFAULT_BATCH_SIZE,
    )
    search_parser.add_argument(
        "--max-length",
        type=_positive_integer,
        default=DEFAULT_MAX_LENGTH,
    )
    search_parser.add_argument("--use-fp16", action="store_true")
    return parser


def _run_index(args: argparse.Namespace) -> int:
    embedder = BGEM3Embedder(
        model_name=args.model,
        use_fp16=args.use_fp16,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    result = build_index(
        input_dir=args.input_dir,
        index_dir=args.index_dir,
        embedder=embedder,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        embedding_options={
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "use_fp16": args.use_fp16,
            "normalized": True,
            "similarity": "dot_product",
        },
    )
    print(
        f"Indice creato: {result.document_count} documenti, "
        f"{result.chunk_count} chunk, {result.embedding_dimensions} dimensioni."
    )
    print(f"File salvati in: {result.index_dir}")
    return 0


def _run_search(args: argparse.Namespace) -> int:
    stored_index = load_index(args.index_dir)
    indexed_model = stored_index.manifest.get("embedding", {}).get("model")
    if not isinstance(indexed_model, str) or not indexed_model:
        raise ValueError("Il manifest non specifica un modello di embedding valido")

    embedder = BGEM3Embedder(
        model_name=args.model or indexed_model,
        use_fp16=args.use_fp16,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    results = search_stored_index(
        stored_index,
        args.query,
        embedder,
        top_k=args.top_k,
    )
    for result in results:
        location = str(result.chunk.source_path)
        if result.chunk.page_number is not None:
            location += f":pagina {result.chunk.page_number}"
        print(f"{result.rank}. score={result.score:.4f} source={location}")
        print(result.chunk.text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Esegue la CLI e restituisce un exit code."""

    parser = create_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "index":
        return _run_index(args)
    if args.command == "search":
        return _run_search(args)
    parser.error(f"Comando sconosciuto: {args.command}")
    return 2
