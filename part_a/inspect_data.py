import pandas as pd
from pathlib import Path

# Location of the generated raw data
DATA_DIR = Path("data/raw")

# Files to inspect
files = [
    "customers.csv",
    "products.csv",
    "orders.csv",
    "tickets.csv",
]

for filename in files:
    file_path = DATA_DIR / filename

    print("\n" + "=" * 60)
    print(f"FILE: {filename}")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print("Rows and columns:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())