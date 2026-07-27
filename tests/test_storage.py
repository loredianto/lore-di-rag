from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from lore_di_rag.chunking import Chunk
from lore_di_rag.storage import load_index, save_index


class StorageTests(unittest.TestCase):
    def test_round_trip_keeps_embeddings_chunks_and_manifest(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            index_dir = Path(temporary_dir)
            chunks = [
                Chunk("primo", Path("documento.txt"), None, 0),
                Chunk("secondo", Path("manuale.pdf"), 2, 0),
            ]
            embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

            save_index(
                index_dir,
                embeddings,
                chunks,
                model_name="fake/model",
                chunk_size=100,
                overlap=10,
                embedding_options={
                    "model": "non-deve-sovrascrivere",
                    "dimensions": 999,
                    "normalized": True,
                },
            )
            stored = load_index(index_dir)

            np.testing.assert_array_equal(stored.embeddings, embeddings)
            self.assertEqual(stored.chunks, chunks)
            self.assertEqual(stored.manifest["embedding"]["model"], "fake/model")
            self.assertEqual(stored.manifest["embedding"]["dimensions"], 2)
            self.assertTrue(stored.manifest["embedding"]["normalized"])

    def test_rejects_embedding_and_chunk_count_mismatch(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, "diverso"):
                save_index(
                    Path(temporary_dir),
                    np.asarray([[1.0, 0.0]], dtype=np.float32),
                    [],
                    model_name="fake/model",
                    chunk_size=100,
                    overlap=10,
                )


if __name__ == "__main__":
    unittest.main()
