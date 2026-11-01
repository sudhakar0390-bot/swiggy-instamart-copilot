
from refund_tools import (
    read_records,
    LEDGER_PATH,
    ESCALATION_PATH,
    issue_refund
)


def main():
    print("PART D - REFUND VERIFICATION")
    print("-" * 40)

    ledger = read_records(LEDGER_PATH)
    escalations = read_records(ESCALATION_PATH)

    print("\nTEST 1: REFUND LEDGER")
    print("Total refund records:", len(ledger))

    for record in ledger:
        print(record)

    print("\nTEST 2: HUMAN ESCALATIONS")
    print("Total escalation records:", len(escalations))

    for record in escalations:
        print(record)

    print("\nTEST 3: DUPLICATE REFUND PROTECTION")
    duplicate = issue_refund("ORD-110227", 75.6)
    print(duplicate)

    print("\nTEST 4: VERIFY SAVED RECORDS")

    refund_found = any(
        record["order_id"] == "ORD-110227"
        and record["refund_inr"] == 75.6
        and record["status"] == "issued"
        for record in ledger
    )

    escalation_found = any(
        record["order_id"] == "ORD-107301"
        and record["refund_inr"] == 1098.0
        and record["status"] == "pending_human_approval"
        for record in escalations
    )

    print("Refund recorded:", refund_found)
    print("Escalation recorded:", escalation_found)

    assert refund_found, "Expected refund record missing"
    assert escalation_found, "Expected escalation record missing"
    assert duplicate.get("error") == "Duplicate refund prevented"

    print("\nALL REFUND VERIFICATION TESTS PASSED")


if __name__ == "__main__":
    main()
