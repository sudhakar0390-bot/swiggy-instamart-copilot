
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm

OUTPUT_DIR = ROOT / "outputs"

SYSTEM_PROMPT = (
    "CLASSIFY support tickets. "
    "Return JSON with category and reason. "
    "Allowed categories: billing, delivery, "
    "technical, fraud."
)


def check_budget(text, max_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]

    estimated_prompt_tokens = sum(
        mock_llm.est_tokens(message["content"])
        for message in messages
    )

    # Reserve tokens for the model's response
    reserved_response_tokens = 100

    if estimated_prompt_tokens + reserved_response_tokens > max_tokens:
        return {
            "status": "blocked_budget",
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "actual_total_tokens": 0
        }

    try:
        response = mock_llm.chat(messages)

        actual_tokens = response.usage.total_tokens

        if actual_tokens > max_tokens:
            status = "exceeded_budget"
        else:
            status = "within_budget"

        return {
            "status": status,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "actual_total_tokens": actual_tokens
        }

    except mock_llm.TransientLLMError:
        return {
            "status": "transient_error",
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "actual_total_tokens": 0
        }


def main():
    mock_llm.reset()

    tickets = pd.read_csv(
        OUTPUT_DIR / "redacted_tickets.csv",
        keep_default_na=False
    )

    # Choose one non-empty ticket
    sample = tickets[
        tickets["redacted_text"].str.strip() != ""
    ].iloc[0]

    text = sample["redacted_text"]

    results = []

    for budget in [20, 100, 500]:
        result = check_budget(text, budget)
        result["budget"] = budget
        results.append(result)

    df = pd.DataFrame(results)

    print("TOKEN BUDGET DEMONSTRATION")
    print("------------------------------")
    print(df.to_string(index=False))

    output_path = OUTPUT_DIR / "triage_budget_demo.csv"
    df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
