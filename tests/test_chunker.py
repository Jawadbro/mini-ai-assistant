"""Unit tests for the chunker — no LLM/network calls needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.chunker import chunk_text


def test_chunk_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_short_text_dropped():
    # Fragments under 20 chars after stripping should be dropped
    assert chunk_text("hi") == []


def test_chunk_produces_overlap():
    text = ("Paragraph one about topic A. " * 40) + "\n\n" + ("Paragraph two about topic B. " * 40)
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) > 20 for c in chunks)


def test_chunk_respects_paragraph_boundary_when_possible():
    text = "First paragraph with enough content to be kept.\n\nSecond paragraph also long enough to keep."
    chunks = chunk_text(text)
    assert len(chunks) >= 1
    joined = " ".join(chunks)
    assert "First paragraph" in joined
    assert "Second paragraph" in joined


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
