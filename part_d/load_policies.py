
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 40
OVERLAP = 10


def chunk_text(text, chunk_size=40, overlap=10):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start + chunk_size >= len(words):
            break

    return chunks


def main():
    records = []

    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    print("PART D - POLICY DOCUMENT CHUNKING")
    print("---------------------------------")
    print("Policy files found:", len(policy_files))

    for policy_file in policy_files:
        text = policy_file.read_text(encoding="utf-8")

        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)

        print(
            policy_file.name,
            "-",
            len(chunks),
            "chunks"
        )

        for index, chunk in enumerate(chunks):
            records.append({
                "document": policy_file.name,
                "chunk_id": f"{policy_file.stem}_{index}",
                "text": chunk
            })

    chunks_df = pd.DataFrame(records)

    output_path = OUTPUT_DIR / "policy_chunks.csv"

    chunks_df.to_csv(output_path, index=False)

    print("\nTotal chunks:", len(chunks_df))
    print("Chunk size:", CHUNK_SIZE)
    print("Overlap:", OVERLAP)

    print("\nFIRST 3 CHUNKS")
    print(chunks_df.head(3).to_string(index=False))

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
