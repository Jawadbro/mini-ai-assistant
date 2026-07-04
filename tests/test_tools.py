"""Unit tests for the mock tools — no LLM/network calls needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools.order_status import get_order_status
from app.tools.product_search import search_product


def test_order_status_found():
    result = get_order_status("ORD001")
    assert result["status"] == "Shipped"
    assert result["estimated_delivery"] == "2026-07-02"


def test_order_status_case_insensitive():
    result = get_order_status("ord001")
    assert result["status"] == "Shipped"


def test_order_status_not_found():
    result = get_order_status("ORD999")
    assert "error" in result


def test_order_status_empty_input():
    result = get_order_status("")
    assert "error" in result


def test_product_search_found():
    result = search_product("wireless mouse")
    assert "matches" in result
    assert result["matches"][0]["name"] == "Wireless Mouse"
    assert result["matches"][0]["price"] == 25
    assert result["matches"][0]["in_stock"] is True


def test_product_search_out_of_stock():
    result = search_product("USB-C Hub")
    assert result["matches"][0]["in_stock"] is False


def test_product_search_partial_match():
    result = search_product("keyboard")
    assert len(result["matches"]) == 1
    assert result["matches"][0]["name"] == "Mechanical Keyboard"


def test_product_search_not_found():
    result = search_product("nonexistent gadget xyz")
    assert "error" in result
    assert result["matches"] == []


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
