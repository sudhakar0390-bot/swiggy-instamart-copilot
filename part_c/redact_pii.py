
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Regular expressions for personal information
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"\bORD-\d{6}\b"


def redact_text(text):
    """Replace personal information with safe placeholders."""

    text = str(text)

    # Count sensitive information before replacing it
    email_count = len(re.findall(EMAIL_PATTERN, text))
    phone_count = len(re.findall(PHONE_PATTERN, text))
    order_count = len(re.findall(ORDER_PATTERN, text))

    # Replace sensitive information
    text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
    text = re.sub(PHONE_PATTERN, "[PHONE]", text)
    text = re.sub(ORDER_PATTERN, "[ORDER_ID]", text)

    return text, email_count, phone_count, order_count


def main():
    tickets = pd.read_csv(
        RAW_DIR / "tickets.csv",
        keep_default_na=False
    )

    results = []

    for _, ticket in tickets.iterrows():
        redacted, emails, phones, orders = redact_text(
            ticket["text"]
        )

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "order_id": ticket["order_id"],
            "redacted_text": redacted,
            "true_category": ticket["true_category"],
            "emails_redacted": emails,
            "phones_redacted": phones,
            "order_ids_redacted": orders
        })

    result_df = pd.DataFrame(results)

    print("PART C - PII REDACTION")
    print("------------------------------")
    print("Total tickets:", len(result_df))

    print("\nREDACTION COUNTS")
    print(
        "Emails:",
        result_df["emails_redacted"].sum()
    )
    print(
        "Phone numbers:",
        result_df["phones_redacted"].sum()
    )
    print(
        "Order IDs:",
        result_df["order_ids_redacted"].sum()
    )

    total = (
        result_df["emails_redacted"].sum()
        + result_df["phones_redacted"].sum()
        + result_df["order_ids_redacted"].sum()
    )

    print("Total PII items redacted:", total)

    print("\nFIRST FIVE REDACTED MESSAGES")
    for _, row in result_df.head(5).iterrows():
        print("\nTicket:", row["ticket_id"])
        print("Redacted:", row["redacted_text"])

    output_path = OUTPUT_DIR / "redacted_tickets.csv"
    result_df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
