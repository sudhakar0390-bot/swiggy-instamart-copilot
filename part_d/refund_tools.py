
from pathlib import Path
import sqlite3
from retrieve_policies import retrieve
import json
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"


def get_order(order_id):
    """Retrieve an order from the SQLite database."""

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT
                order_id,
                customer_id,
                status,
                order_amount,
                delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

    if row is None:
        return {
            "error": "Order not found",
            "order_id": order_id
        }

    return dict(row)

def search_policy(query):
    """Retrieve the top 3 relevant policy chunks."""

    results = retrieve(query, top_k=3)

    return {
        "query": query,
        "results": results
    }


def calculate_refund(order_amount, delivery_minutes):
    """Calculate a refund for a completed, eligible order."""

    if order_amount is None or delivery_minutes is None:
        return {
            "error": "Missing order amount or delivery time"
        }

    order_amount = float(order_amount)
    delivery_minutes = float(delivery_minutes)

    if order_amount < 0 or delivery_minutes < 0:
        return {
            "error": "Invalid order amount or delivery time"
        }

    # Refund policy: delivery taking more than 45 minutes
    if delivery_minutes > 45:
        refund_amount = round(order_amount, 2)
    else:
        refund_amount = 0.0

    return {
        "refund_inr": refund_amount,
        "delivery_minutes": delivery_minutes,
        "eligible": refund_amount > 0
    }


LEDGER_PATH = ROOT / "outputs" / "refund_ledger.jsonl"
ESCALATION_PATH = ROOT / "outputs" / "refund_escalations.jsonl"


def read_records(path):
    """Read existing JSONL records."""
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def issue_refund(order_id, refund_inr):
    """Record a simulated refund with duplicate protection."""

    order = get_order(order_id)

    if "error" in order:
        return {"error": "Order not found"}

    if order["status"] != "completed":
        return {"error": "Only completed orders can be refunded"}

    calculation = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"]
    )

    expected_amount = calculation.get("refund_inr", 0)

    if expected_amount <= 0:
        return {"error": "Order is not eligible for a refund"}

    if expected_amount > 1000:
        return {"error": "Human approval required"}

    if (
        isinstance(refund_inr, bool)
        or not isinstance(refund_inr, (int, float))
        or round(refund_inr, 2) != expected_amount
    ):
        return {"error": "Refund amount does not match policy"}

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = read_records(LEDGER_PATH)

    for record in existing:
        if record["order_id"] == order_id:
            return {
                "error": "Duplicate refund prevented",
                "order_id": order_id
            }

    record = {
        "order_id": order_id,
        "refund_inr": expected_amount,
        "status": "issued"
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return record


def escalate_to_human(order_id, refund_inr):
    """Record a refund request requiring human approval."""

    order = get_order(order_id)

    if "error" in order:
        return {"error": "Order not found"}

    if order["status"] != "completed":
        return {"error": "Only completed orders can be refunded"}

    calculation = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"]
    )

    expected_amount = calculation.get("refund_inr", 0)

    if expected_amount <= 1000:
        return {"error": "Human approval is not required"}

    if (
        isinstance(refund_inr, bool)
        or not isinstance(refund_inr, (int, float))
        or round(refund_inr, 2) != expected_amount
    ):
        return {"error": "Refund amount does not match policy"}

    ESCALATION_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = read_records(ESCALATION_PATH)

    for record in existing:
        if record["order_id"] == order_id:
            return {
                "status": "already_escalated",
                "order_id": order_id,
                "refund_inr": expected_amount
            }

    record = {
        "order_id": order_id,
        "refund_inr": expected_amount,
        "status": "pending_human_approval"
    }

    with open(ESCALATION_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return record


def main():
    print("PART D - REFUND TOOLS")
    print("-----------------------------")

    print("\nTEST 1: GET ORDER")
    print(get_order("ORD-108250"))

    print("\nTEST 2: SEARCH POLICY")
    policy = search_policy("refund policy late delivery")

    for result in policy["results"]:
        print(
            result["document"],
            "| Score:",
            result["score"]
        )

    print("\nTEST 3: LATE DELIVERY")
    print(calculate_refund(250, 60))

    print("\nTEST 4: ON-TIME DELIVERY")
    print(calculate_refund(250, 30))

    print("\nTEST 5: REFUND THRESHOLD")
    print(calculate_refund(250, 45))


if __name__ == "__main__":
    main()



    print("\nTEST 6: PREVENT INELIGIBLE REFUND")
    print(issue_refund("ORD-108250", 310.0))

    print("\nTEST 7: PREVENT INVALID ESCALATION")
    print(escalate_to_human("ORD-108250", 310.0))
