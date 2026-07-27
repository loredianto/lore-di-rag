from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from lore_di_rag.ingestion import discover_input_files, load_documents


class DocumentLoaderTests(unittest.TestCase):
    def test_discovers_supported_files_recursively(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            nested_dir = input_dir / "nested"
            nested_dir.mkdir()
            (input_dir / "notes.txt").write_text("Primo testo", encoding="utf-8")
            (nested_dir / "guide.md").write_text("# Guida", encoding="utf-8")
            (input_dir / "ignored.csv").write_text("a,b", encoding="utf-8")

            files = discover_input_files(input_dir)

            self.assertEqual(
                files,
                [input_dir / "nested" / "guide.md", input_dir / "notes.txt"],
            )

    def test_loads_non_empty_text_documents(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            input_dir = Path(temporary_dir)
            source_path = input_dir / "document.txt"
            source_path.write_text("  Contenuto da indicizzare.  ", encoding="utf-8")
            (input_dir / "empty.md").write_text("  ", encoding="utf-8")

            documents = load_documents(input_dir)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].text, "Contenuto da indicizzare.")
            self.assertEqual(documents[0].source_path, source_path)
            self.assertIsNone(documents[0].page_number)

    def test_rejects_a_missing_input_directory(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            missing_dir = Path(temporary_dir) / "missing"

            with self.assertRaises(FileNotFoundError):
                discover_input_files(missing_dir)


if __name__ == "__main__":
    unittest.main()
