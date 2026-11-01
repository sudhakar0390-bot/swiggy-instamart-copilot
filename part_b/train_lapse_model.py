
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

data = pd.read_csv(OUTPUT_DIR / "lapse_dataset.csv")

# Features use only information available before cutoff.
numeric_features = [
    "age",
    "frequency",
    "monetary",
    "avg_discount",
    "avg_quantity",
    "recency"
]

categorical_features = ["city", "gender"]

feature_columns = numeric_features + categorical_features

X = data[feature_columns]
y = data["lapsed"]

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training customers:", len(X_train))
print("Testing customers:", len(X_test))

# Numeric preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(
        strategy="most_frequent"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Model 1: Logistic Regression
logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

logistic_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

# Model 2: Random Forest
forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

forest_grid = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20]
}

models = {
    "Logistic Regression": (
        logistic_pipeline,
        logistic_grid
    ),
    "Random Forest": (
        forest_pipeline,
        forest_grid
    )
}

results = []

for name, (pipeline, params) in models.items():

    print(f"\nTraining {name}...")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    predictions = search.predict(X_test)
    test_f1 = f1_score(y_test, predictions)

    print("Best parameters:", search.best_params_)
    print("Cross-validation F1:", round(search.best_score_, 4))
    print("Test F1:", round(test_f1, 4))

    results.append({
        "name": name,
        "model": search.best_estimator_,
        "cv_f1": search.best_score_,
        "test_f1": test_f1
    })

# Select by cross-validation performance,
# not by looking at test scores.
best = max(results, key=lambda r: r["cv_f1"])

print("\nSELECTED MODEL:", best["name"])

final_predictions = best["model"].predict(X_test)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    final_predictions,
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, final_predictions))

# Save the selected model
joblib.dump(
    best["model"],
    OUTPUT_DIR / "lapse_model.joblib"
)

print("\nSaved: outputs/lapse_model.joblib")

# Save test predictions for retention analysis
test_results = data.loc[X_test.index, [
    "customer_id", "lapsed"
]].copy()

test_results["predicted_lapsed"] = final_predictions
test_results["lapse_probability"] = (
    best["model"].predict_proba(X_test)[:, 1]
)

test_results.to_csv(
    OUTPUT_DIR / "lapse_test_predictions.csv",
    index=False
)

print("Saved: outputs/lapse_test_predictions.csv")
