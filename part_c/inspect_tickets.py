
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# Load customer support tickets
tickets = pd.read_csv(
    RAW_DIR / "tickets.csv",
    keep_default_na=False
)

print("PART C - SUPPORT TICKET INSPECTION")
print("--------------------------------")

print("\nTotal tickets:", len(tickets))

print("\nCOLUMN NAMES")
print(tickets.columns.tolist())

print("\nFIRST FIVE TICKETS")
print(tickets.head(5).to_string(index=False))

print("\nMISSING OR EMPTY VALUES")
for column in tickets.columns:
    empty_count = (
        tickets[column].astype(str).str.strip() == ""
    ).sum()

    print(column, ":", empty_count)

print("\nDUPLICATE ROWS")
print(tickets.duplicated().sum())

print("\nDATA TYPES")
print(tickets.dtypes)
