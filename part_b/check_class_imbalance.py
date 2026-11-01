
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

# Load prepared lapse dataset
data = pd.read_csv(OUTPUT_DIR / "lapse_dataset.csv")

features = [
    "age",
    "frequency",
    "monetary",
    "avg_discount",
    "avg_quantity",
    "recency",
    "city",
    "gender"
]

X = data[features]
y = data["lapsed"]

# Reproduce the original train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("ORIGINAL TRAINING CLASS COUNTS")
print(y_train.value_counts().sort_index())

# Load the previously selected model
original_model = joblib.load(
    OUTPUT_DIR / "lapse_model.joblib"
)

original_predictions = original_model.predict(X_test)
original_f1 = f1_score(y_test, original_predictions)

# Combine training features and labels
train_data = X_train.copy()
train_data["lapsed"] = y_train

# Separate majority and minority classes
majority = train_data[train_data["lapsed"] == 0]
minority = train_data[train_data["lapsed"] == 1]

# Oversample minority class using training rows only
minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

balanced_train = pd.concat(
    [majority, minority_upsampled],
    ignore_index=True
)

# Shuffle balanced training data
balanced_train = balanced_train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

X_balanced = balanced_train[features]
y_balanced = balanced_train["lapsed"]

print("\nBALANCED TRAINING CLASS COUNTS")
print(y_balanced.value_counts().sort_index())

# Clone the selected model so the original stays unchanged
from sklearn.base import clone

balanced_model = clone(original_model)
balanced_model.fit(X_balanced, y_balanced)

balanced_predictions = balanced_model.predict(X_test)
balanced_f1 = f1_score(y_test, balanced_predictions)

print("\nF1 SCORE COMPARISON")
print("Original model F1:", round(original_f1, 4))
print("Oversampled model F1:", round(balanced_f1, 4))

print("\nOVERSAMPLED MODEL CLASSIFICATION REPORT")
print(classification_report(
    y_test,
    balanced_predictions,
    zero_division=0
))

if balanced_f1 > original_f1:
    print("\nOversampling improved the test F1 score.")
else:
    print("\nOversampling did not improve the test F1 score.")

print("\nOriginal saved model was not modified.")
