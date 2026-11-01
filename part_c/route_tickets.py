
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

# Load classified tickets and customer segments
tickets = pd.read_csv(
    OUTPUT_DIR / "triage_results.csv"
)

segments = pd.read_csv(
    OUTPUT_DIR / "customer_segments.csv"
)

# Add customer segment to each ticket
routed = tickets.merge(
    segments[["customer_id", "segment"]],
    on="customer_id",
    how="left",
    validate="many_to_one"
)


def assign_route(row):
    category = row["predicted_category"]
    segment = row["segment"]

    if pd.isna(category):
        return "Review", "Manual Review"

    if category == "fraud":
        return "Critical", "Fraud Investigation"

    if segment == "Premium":
        return "High", "Priority Support"

    if category == "billing":
        return "Medium", "Billing Support"

    if category == "technical":
        return "Medium", "Technical Support"

    if category == "delivery":
        return "Medium", "Delivery Support"

    return "Normal", "General Support"


# Apply routing rules
routes = routed.apply(assign_route, axis=1)

routed["priority"] = routes.apply(lambda x: x[0])
routed["assigned_queue"] = routes.apply(lambda x: x[1])

print("PART C - TICKET ROUTING")
print("--------------------------------")

print("Total tickets:", len(routed))

print("\nCUSTOMER SEGMENTS")
print(routed["segment"].value_counts(dropna=False))

print("\nPRIORITY DISTRIBUTION")
print(routed["priority"].value_counts())

print("\nSUPPORT QUEUES")
print(routed["assigned_queue"].value_counts())

print("\nFIRST 10 ROUTED TICKETS")
print(
    routed[[
        "ticket_id",
        "customer_id",
        "segment",
        "predicted_category",
        "priority",
        "assigned_queue"
    ]].head(10).to_string(index=False)
)

output_path = OUTPUT_DIR / "routed_tickets.csv"
routed.to_csv(output_path, index=False)

print("\nSaved:", output_path)
