

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm
from refund_agent import run_agent


def main():
    print("PART D - AGENT SAFETY TESTS")
    print("-" * 40)

    # TEST 1: Maximum iterations
    mock_llm.reset()

    result1 = run_agent(
        "Refund order ORD-108250 [LOOP]",
        max_iterations=6
    )

    print("\nTEST 1: MAXIMUM ITERATIONS")
    print("Result:", result1)

    assert result1 == "FALLBACK_MAX_ITERATIONS"

    # TEST 2: Token limit
    mock_llm.reset()

    result2 = run_agent(
    "Refund order ORD-108250",
    token_limit=1

    )

    print("\nTEST 2: TOKEN LIMIT")
    print("Result:", result2)

    assert result2 == "FALLBACK_TOKEN_LIMIT"

    # TEST 3: Timeout
    mock_llm.reset()

    result3 = run_agent(
        "Refund order ORD-108250",
        timeout_seconds=0
    )

    print("\nTEST 3: TIMEOUT")
    print("Result:", result3)

    assert result3 == "FALLBACK_TIMEOUT"

    print("\nALL AGENT SAFETY TESTS PASSED")


if __name__ == "__main__":
    main()
