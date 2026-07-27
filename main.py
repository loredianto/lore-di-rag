"""Launcher di compatibilità; l'applicazione vive nel package ``lore_di_rag``."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lore_di_rag.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
