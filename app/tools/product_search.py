"""Mock 'Product Search' tool backed by data/products.json."""
import json
from app.config import PRODUCTS_FILE

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_product",
        "description": (
            "Search the product catalog by (partial) product name and "
            "return its price and stock availability. Use this whenever "
            "the user asks if a product is available, its price, or "
            "stock, e.g. 'do you have a wireless mouse?'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "Product name or partial name to search for, e.g. 'wireless mouse'",
                }
            },
            "required": ["product_name"],
        },
    },
}


def _load_products() -> list[dict]:
    if not PRODUCTS_FILE.exists():
        return []
    with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def search_product(product_name: str) -> dict:
    """Case-insensitive substring match against the product catalog."""
    if not product_name or not product_name.strip():
        return {"error": "No product name provided."}

    products = _load_products()
    query = product_name.strip().lower()

    matches = [p for p in products if query in p["name"].lower()]

    if not matches:
        return {"error": f"No products found matching '{product_name}'.", "matches": []}

    return {
        "matches": [
            {
                "name": p["name"],
                "price": p["price"],
                "stock": p["stock"],
                "in_stock": p["stock"] > 0,
            }
            for p in matches
        ]
    }
