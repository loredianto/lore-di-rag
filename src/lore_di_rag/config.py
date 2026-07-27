"""Configurazione e percorsi predefiniti del progetto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 512


@dataclass(frozen=True)
class ProjectPaths:
    """Percorsi convenzionali, calcolati rispetto alla radice del progetto."""

    root: Path
    input_dir: Path
    index_dir: Path
    plots_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        resolved_root = root.expanduser().resolve()
        return cls(
            root=resolved_root,
            input_dir=resolved_root / "data" / "input",
            index_dir=resolved_root / "data" / "indexes" / "default",
            plots_dir=resolved_root / "output" / "plots",
        )


DEFAULT_PATHS = ProjectPaths.from_root(Path.cwd())
