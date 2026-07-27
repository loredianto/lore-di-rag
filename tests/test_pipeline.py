from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lore_di_rag.pipeline import build_index
from lore_di_rag.storage import load_index


class FakeEmbedder:
    model_name = "fake/test-embedder"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[float(len(text)), float(index)] for index, text in enumerate(texts)],
            dtype=np.float32,
        )


class BuildIndexTests(unittest.TestCase):
    def test_builds_npy_index_with_aligned_chunk_metadata(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            index_dir = root / "index"
            input_dir.mkdir()
            (input_dir / "note.txt").write_text(
                "Primo paragrafo.\n\nSecondo paragrafo abbastanza lungo.",
                encoding="utf-8",
            )

            result = build_index(
                input_dir=input_dir,
                index_dir=index_dir,
                embedder=FakeEmbedder(),
                chunk_size=30,
                overlap=5,
            )
            stored = load_index(index_dir)

            self.assertTrue((index_dir / "embeddings.npy").is_file())
            self.assertTrue((index_dir / "chunks.jsonl").is_file())
            self.assertTrue((index_dir / "manifest.json").is_file())
            self.assertEqual(result.document_count, 1)
            self.assertEqual(result.chunk_count, len(stored.chunks))
            self.assertEqual(stored.embeddings.shape, (len(stored.chunks), 2))
            self.assertEqual(stored.manifest["embedding"]["model"], FakeEmbedder.model_name)

    def test_rejects_an_input_directory_without_supported_documents(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_dir = root / "input"
            input_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "Nessun documento"):
                build_index(
                    input_dir=input_dir,
                    index_dir=root / "index",
                    embedder=FakeEmbedder(),
                )


if __name__ == "__main__":
    unittest.main()
