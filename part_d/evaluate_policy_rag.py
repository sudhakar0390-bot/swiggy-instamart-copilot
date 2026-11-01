
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "part_d"))

from retrieve_policies import retrieve
from policy_rag import answer_policy_question

import mock_llm

EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"


def main():
    mock_llm.reset()

    evaluation = pd.read_csv(
        EVAL_FILE,
        keep_default_na=False
    )

    results = []

    for _, row in evaluation.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        # Retrieve the top 3 chunks
        retrieved = retrieve(question, top_k=3)

        retrieved_docs = [
            item["document"] for item in retrieved
        ]

        # Generate the policy answer
        response = answer_policy_question(question)
        answer = response["answer"]

        # Check retrieval
        if expected_doc == "NONE":
            retrieval_ok = None
        else:
            retrieval_ok = expected_doc in retrieved_docs

        # Check answer
        if expected_doc == "NONE":
            fallback_phrases = [
                "could not find",
                "not found",
                "not available",
                "cannot answer",
                "don't know",
                "do not know"
            ]

            fact_ok = any(
                phrase in answer.lower()
                for phrase in fallback_phrases
            )

        else:
            fact_ok = (
                expected_fact.lower()
                in answer.lower()
            )

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "retrieved_docs": ", ".join(retrieved_docs),
            "retrieval_ok": retrieval_ok,
            "answer": answer,
            "fact_ok": fact_ok
        })

    results_df = pd.DataFrame(results)

    supported = results_df[
        results_df["expected_doc"] != "NONE"
    ]

    unsupported = results_df[
        results_df["expected_doc"] == "NONE"
    ]

    print("PART D - POLICY RAG EVALUATION")
    print("--------------------------------")

    print("Total questions:", len(results_df))

    print(
        "Retrieval success:",
        int(supported["retrieval_ok"].sum()),
        "/",
        len(supported)
    )

    print(
        "Fact success:",
        int(supported["fact_ok"].sum()),
        "/",
        len(supported)
    )

    print(
        "Fallback success:",
        int(unsupported["fact_ok"].sum()),
        "/",
        len(unsupported)
    )

    print("\nQUESTION RESULTS")

    for _, row in results_df.iterrows():
        print("\nQUESTION:", row["question"])
        print("EXPECTED FACT:", row["expected_fact"])
        print("ANSWER:", row["answer"])
        print("FACT MATCH:", row["fact_ok"])

    output_path = OUTPUT_DIR / "policy_eval_results.csv"

    results_df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
