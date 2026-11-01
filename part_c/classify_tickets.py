
from pathlib import Path
import sys
import json
import re
import time

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"

# Allow importing mock_llm.py from project root
sys.path.insert(0, str(ROOT))

import mock_llm

ALLOWED_CATEGORIES = {
    "billing", "delivery", "technical", "fraud"
}

KEYWORDS = {
    "fraud": {"unauthorized", "fraud", "hacked", "stolen"},
    "technical": {"app", "crash", "login", "otp", "error", "freezes"},
    "billing": {"charged", "invoice", "bill", "refund", "deducted", "payment"},
    "delivery": {"arrived", "rider", "delivery", "delivered", "late"}
}

MAX_TOKENS = 500


def keyword_fallback(text):
    words = set(re.findall(r"[a-z]+", text.lower()))

    for category in [
        "fraud", "technical", "billing", "delivery"
    ]:
        if KEYWORDS[category] & words:
            return category

    return None


def classify_ticket(text):
    if not text.strip():
        return None, "flagged_empty", 0

    estimated_tokens = mock_llm.est_tokens(text)

    if estimated_tokens > MAX_TOKENS:
        return None, "flagged_long", 0

    messages = [
        {
            "role": "system",
            "content": (
                "CLASSIFY support tickets. "
                "Return JSON with category and reason. "
                "Allowed categories: billing, delivery, "
                "technical, fraud."
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]

    response = None

    for attempt in range(3):
        try:
            response = mock_llm.chat(messages)
            break

        except mock_llm.TransientLLMError:
            if attempt == 2:
                return None, "flagged_llm_failure", 0

            time.sleep(0.1 * (2 ** attempt))

    if response is None:
        return None, "flagged_llm_failure", 0

    used_tokens = response.usage.total_tokens
    content = response.choices[0].message.content

    try:
        parsed = json.loads(content)

        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object")

        category = parsed.get("category")

        if category not in ALLOWED_CATEGORIES:
            raise ValueError("Invalid category")

        return category, "classified_json", used_tokens

    
    except json.JSONDecodeError:
        # Non-JSON response: try keyword fallback
        fallback = keyword_fallback(text)

        if fallback is not None:
            return fallback, "classified_keyword", used_tokens

        return None, "flagged_invalid_output", used_tokens

    except (ValueError, TypeError):
        # Invalid JSON structure or unsupported category:
        # flag for human review
        return None, "flagged_invalid_output", used_tokens



def main():
    mock_llm.reset()

    tickets = pd.read_csv(
        OUTPUT_DIR / "redacted_tickets.csv",
        keep_default_na=False
    )

    results = []

    for _, row in tickets.iterrows():
        text = row["redacted_text"]

        category, status, tokens = classify_ticket(text)

        results.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "predicted_category": category,
            "status": status,
            "tokens_used": tokens
        })

    results_df = pd.DataFrame(results)

    print("PART C - TICKET CLASSIFICATION")
    print("--------------------------------")

    print("Total tickets:", len(results_df))

    print("\nCLASSIFICATION STATUS")
    print(results_df["status"].value_counts())

    print("\nPREDICTED CATEGORIES")
    print(results_df["predicted_category"].value_counts())

    # Accuracy only for successfully classified tickets
    classified = results_df[
        results_df["predicted_category"].notna()
    ].copy()

    if len(classified):
        accuracy = (
            classified["predicted_category"]
            == classified["true_category"]
        ).mean()

        print("\nClassified tickets:", len(classified))
        print("Classification accuracy:", round(accuracy, 4))

    print("\nTOTAL TOKENS USED")
    print(results_df["tokens_used"].sum())

    output_path = OUTPUT_DIR / "triage_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()

