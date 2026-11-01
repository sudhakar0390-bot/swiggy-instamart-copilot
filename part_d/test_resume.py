
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part_d"))

import mock_llm
from refund_agent import run_agent
from checkpoint import load_checkpoint, clear_checkpoint


def main():
    request = "Refund order ORD-108250"

    mock_llm.reset()
    clear_checkpoint()

    print("TEST 1: Simulate interruption")

    result1 = run_agent(
        request,
        max_iterations=1
    )

    print("First result:", result1)

    checkpoint = load_checkpoint()

    assert result1 == "FALLBACK_MAX_ITERATIONS"
    assert checkpoint is not None
    assert checkpoint["iteration"] == 1

    print("Saved iteration:", checkpoint["iteration"])

    print("\nTEST 2: Resume saved workflow")

    result2 = run_agent(
        request,
        resume=True,
        max_iterations=6
    )

    print("Resumed result:", result2)

    assert result2 == "DENIED: delivery was within 45 minutes."
    assert load_checkpoint() is None

    print("\nALL CHECKPOINT RESUME TESTS PASSED")


if __name__ == "__main__":
    main()
