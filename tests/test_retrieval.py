from pathlib import Path
import unittest

import numpy as np

from lore_di_rag.chunking import Chunk
from lore_di_rag.retrieval import rank_chunks, search_stored_index
from lore_di_rag.storage import StoredIndex


def make_index() -> StoredIndex:
    return StoredIndex(
        embeddings=np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
        chunks=[
            Chunk("primo", Path("a.txt"), None, 0),
            Chunk("secondo", Path("b.txt"), None, 0),
            Chunk("terzo", Path("c.txt"), None, 0),
        ],
        manifest={"embedding": {"model": "fake/model", "dimensions": 2}},
    )


class FakeEmbedder:
    model_name = "fake/model"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[0.0, 1.0]], dtype=np.float32)


class RetrievalTests(unittest.TestCase):
    def test_ranks_by_dot_product_with_stable_order(self) -> None:
        results = rank_chunks(
            make_index(),
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            top_k=2,
        )

        self.assertEqual([result.chunk.text for result in results], ["primo", "terzo"])
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual(results[0].score, 1.0)

    def test_search_encodes_query_without_storing_it(self) -> None:
        results = search_stored_index(
            make_index(),
            "  una query  ",
            FakeEmbedder(),
            top_k=1,
        )

        self.assertEqual(results[0].chunk.text, "secondo")

    def test_rejects_model_or_dimension_mismatch(self) -> None:
        wrong_model = FakeEmbedder()
        wrong_model.model_name = "other/model"
        with self.assertRaisesRegex(ValueError, "non coincide"):
            search_stored_index(make_index(), "query", wrong_model)

        with self.assertRaisesRegex(ValueError, "Dimensione query"):
            rank_chunks(
                make_index(),
                np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            )

    def test_rejects_empty_query_and_invalid_top_k(self) -> None:
        with self.assertRaisesRegex(ValueError, "non può essere vuota"):
            search_stored_index(make_index(), " ", FakeEmbedder())
        with self.assertRaisesRegex(ValueError, "top_k"):
            rank_chunks(
                make_index(),
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                top_k=0,
            )

    def test_rejects_an_index_not_aligned_with_its_chunks(self) -> None:
        index = make_index()
        invalid_index = StoredIndex(
            embeddings=index.embeddings[:1],
            chunks=index.chunks,
            manifest=index.manifest,
        )

        with self.assertRaisesRegex(ValueError, "non è allineata"):
            rank_chunks(
                invalid_index,
                np.asarray([[1.0, 0.0]], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
