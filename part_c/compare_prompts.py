
from pathlib import Path
import sys
import json
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm

OUTPUT_DIR = ROOT / "outputs"

CATEGORIES = {"billing", "delivery", "technical", "fraud"}

SYSTEM_ZERO = (
    "CLASSIFY support tickets. "
    "Return JSON with category and reason. "
    "Allowed categories: billing, delivery, technical, fraud."
)

SYSTEM_FEW = (
    "CLASSIFY support tickets. "
    "Return JSON with category and reason. "
    "Allowed categories: billing, delivery, technical, fraud.\n"
    "EXAMPLES:\n"
    "Unauthorized transaction -> fraud\n"
    "App crashes during checkout -> technical\n"
    "Wrong invoice amount -> billing\n"
    "Rider arrived late -> delivery"
)

KEYWORDS = {
    "fraud": {"unauthorized", "fraud", "hacked", "stolen"},
    "technical": {"app", "crash", "login", "otp", "error", "freezes"},
    "billing": {"charged", "invoice", "bill", "refund", "deducted", "payment"},
    "delivery": {"arrived", "rider", "delivery", "delivered", "late"}
}


def keyword_fallback(text, few_shot=False):
    words = set(re.findall(r"[a-z]+", text.lower()))

    if few_shot:
        order = ["fraud", "technical", "billing", "delivery"]
    else:
        order = ["billing", "fraud", "technical", "delivery"]

    for category in order:
        if KEYWORDS[category] & words:
            return category

    return None


def run_experiment(tickets, system_prompt, few_shot):
    mock_llm.reset()

    predictions = []

    for _, row in tickets.iterrows():
        text = str(row["redacted_text"])

        if not text.strip() or mock_llm.est_tokens(text) > 500:
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        response = None

        for attempt in range(3):
            try:
                response = mock_llm.chat(messages)
                break
            except mock_llm.TransientLLMError:
                if attempt == 2:
                    break

        if response is None:
            continue

        content = response.choices[0].message.content

        try:
            parsed = json.loads(content)

            if not isinstance(parsed, dict):
                continue

            category = parsed.get("category")

            if category not in CATEGORIES:
                continue

        except (json.JSONDecodeError, TypeError):
            category = keyword_fallback(text, few_shot)

        if category not in CATEGORIES:
            continue

        predictions.append({
            "ticket_id": row["ticket_id"],
            "true_category": row["true_category"],
            "predicted_category": category,
            "correct": category == row["true_category"]
        })

    result = pd.DataFrame(predictions)

    accuracy = result["correct"].mean() if len(result) else 0

    return len(result), accuracy


def main():
    tickets = pd.read_csv(
        OUTPUT_DIR / "redacted_tickets.csv",
        keep_default_na=False
    )

    zero_count, zero_accuracy = run_experiment(
        tickets, SYSTEM_ZERO, False
    )

    few_count, few_accuracy = run_experiment(
        tickets, SYSTEM_FEW, True
    )

    print("ZERO-SHOT VS FEW-SHOT COMPARISON")
    print("--------------------------------")
    print(
        "Zero-shot classified:",
        zero_count,
        "Accuracy:",
        round(zero_accuracy, 4)
    )

    print(
        "Few-shot classified:",
        few_count,
        "Accuracy:",
        round(few_accuracy, 4)
    )

    print(
        "Accuracy improvement:",
        round(few_accuracy - zero_accuracy, 4)
    )


if __name__ == "__main__":
    main()
