from pathlib import Path
import unittest

from lore_di_rag.chunking import chunk_documents, chunk_text
from lore_di_rag.ingestion import LoadedDocument


class ChunkingTests(unittest.TestCase):
    def test_overlaps_consecutive_chunks(self) -> None:
        chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)

        self.assertEqual(chunks, ["abcdef", "efghij"])

    def test_preserves_source_page_and_chunk_order(self) -> None:
        source_path = Path("manuale.pdf")
        chunks = chunk_documents(
            [LoadedDocument("abcdefghij", source_path, page_number=3)],
            chunk_size=6,
            overlap=2,
        )

        self.assertEqual([chunk.chunk_index for chunk in chunks], [0, 1])
        self.assertTrue(all(chunk.source_path == source_path for chunk in chunks))
        self.assertTrue(all(chunk.page_number == 3 for chunk in chunks))

    def test_rejects_invalid_window_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            chunk_text("testo", chunk_size=0)
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_text("testo", chunk_size=10, overlap=10)


if __name__ == "__main__":
    unittest.main()
