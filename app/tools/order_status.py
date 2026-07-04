"""Mock 'Order Status' tool backed by data/orders.json."""
import json
from app.config import ORDERS_FILE

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": (
            "Look up the status and estimated delivery date of a customer "
            "order by its order ID. Use this whenever the user asks about "
            "an order, shipment, or delivery status, e.g. 'where is my "
            "order ORD001?'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID, e.g. 'ORD001'",
                }
            },
            "required": ["order_id"],
        },
    },
}


def _load_orders() -> list[dict]:
    if not ORDERS_FILE.exists():
        return []
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_order_status(order_id: str) -> dict:
    """Returns order status dict, or an error payload if not found."""
    if not order_id or not order_id.strip():
        return {"error": "No order ID provided."}

    orders = _load_orders()
    normalized = order_id.strip().upper()

    for order in orders:
        if order.get("order_id", "").upper() == normalized:
            return {
                "order_id": order["order_id"],
                "status": order["status"],
                "estimated_delivery": order["estimated_delivery"],
            }

    return {"error": f"No order found with ID '{order_id}'."}
