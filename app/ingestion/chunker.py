"""
Splits raw document text into overlapping chunks suitable for embedding.

Uses a recursive character splitter: tries to break on paragraph boundaries
first, then sentences, then words, only falling back to a hard character
cut as a last resort. This keeps chunks semantically coherent instead of
slicing mid-sentence.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    # Drop near-empty fragments (e.g. stray whitespace chunks from PDFs)
    return [c.strip() for c in chunks if len(c.strip()) > 20]
