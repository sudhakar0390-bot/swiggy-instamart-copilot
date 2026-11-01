
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

# Load customer segments
segments = pd.read_csv(
    OUTPUT_DIR / "customer_segments.csv"
)

# Load test-set predictions
predictions = pd.read_csv(
    OUTPUT_DIR / "lapse_test_predictions.csv"
)

# Join using customer ID
retention = predictions.merge(
    segments[[
        "customer_id",
        "segment",
        "recency",
        "frequency",
        "monetary"
    ]],
    on="customer_id",
    how="left",
    validate="one_to_one"
)

# Keep Premium customers only
premium = retention[
    retention["segment"] == "Premium"
].copy()

# Rank by predicted lapse probability
premium = premium.sort_values(
    "lapse_probability",
    ascending=False
)

# Select top 10
top_10 = premium.head(10).copy()

# Add ranking
top_10.insert(
    0,
    "rank",
    range(1, len(top_10) + 1)
)

# Save result
output_path = OUTPUT_DIR / "retention_list.csv"
top_10.to_csv(output_path, index=False)

print("PREMIUM CUSTOMER RETENTION LIST")
print("--------------------------------")
print("Premium customers in test set:", len(premium))

print("\nTOP 10 CUSTOMERS")
print(top_10.to_string(index=False))

print("\nSaved:", output_path)
