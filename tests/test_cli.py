from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from lore_di_rag import cli
from lore_di_rag.chunking import Chunk
from lore_di_rag.storage import save_index


class FakeEmbedder:
    model_name = "fake/model"

    def __init__(self, **options: object) -> None:
        self.options = options
        if isinstance(options.get("model_name"), str):
            self.model_name = str(options["model_name"])

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class CliTests(unittest.TestCase):
    def test_no_command_prints_help(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            exit_code = cli.main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("index", output.getvalue())
        self.assertIn("search", output.getvalue())

    def test_index_builds_files_with_a_fake_embedder(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            index_dir = root / "index"
            input_dir.mkdir()
            (input_dir / "note.txt").write_text(
                "Testo da indicizzare.",
                encoding="utf-8",
            )
            output = StringIO()
            with (
                patch("lore_di_rag.cli.BGEM3Embedder", FakeEmbedder),
                redirect_stdout(output),
            ):
                exit_code = cli.main(
                    [
                        "index",
                        "--input-dir",
                        str(input_dir),
                        "--index-dir",
                        str(index_dir),
                        "--model",
                        "fake/model",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((index_dir / "embeddings.npy").is_file())
            self.assertTrue((index_dir / "chunks.jsonl").is_file())
            self.assertTrue((index_dir / "manifest.json").is_file())
            self.assertIn("Indice creato", output.getvalue())

    def test_search_uses_model_from_manifest_without_downloading_it(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            index_dir = Path(temporary_dir)
            save_index(
                index_dir,
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                [Chunk("contenuto", Path("document.txt"), None, 0)],
                model_name="fake/model",
                chunk_size=100,
                overlap=10,
            )
            output = StringIO()
            with (
                patch("lore_di_rag.cli.BGEM3Embedder", FakeEmbedder),
                redirect_stdout(output),
            ):
                exit_code = cli.main(
                    ["search", "domanda", "--index-dir", str(index_dir)]
                )

        self.assertEqual(exit_code, 0)
        self.assertIn("contenuto", output.getvalue())
        self.assertIn("score=1.0000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
