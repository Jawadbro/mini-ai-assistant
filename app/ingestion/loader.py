"""
Loads raw text out of an uploaded file.

Supports .pdf, .txt, .md. Each loader returns a single string of the
document's full text; page/line structure is discarded here on purpose —
the chunker is what imposes structure we actually use downstream.
"""
from pathlib import Path
from pypdf import PdfReader


class UnsupportedFileType(Exception):
    pass


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_document(path: Path) -> str:
    """Dispatch on file extension. Raises UnsupportedFileType otherwise."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in (".txt", ".md", ".markdown"):
        return load_text(path)
    raise UnsupportedFileType(
        f"Unsupported file type '{suffix}'. Supported: .pdf, .txt, .md"
    )
