
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "state"
CHECKPOINT_PATH = CHECKPOINT_DIR / "refund_agent_checkpoint.json"


def save_checkpoint(messages, iteration, total_tokens, trace):
    """Save agent progress to disk."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "messages": messages,
        "iteration": iteration,
        "total_tokens": total_tokens,
        "trace": trace
    }

    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as file:
        json.dump(
    data,
    file,
    indent=2,
    default=lambda obj: vars(obj)
)

    print("Checkpoint saved:", CHECKPOINT_PATH)


def load_checkpoint():
    """Load previously saved agent progress."""
    if not CHECKPOINT_PATH.exists():
        return None

    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def clear_checkpoint():
    """Remove checkpoint after successful completion."""
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
