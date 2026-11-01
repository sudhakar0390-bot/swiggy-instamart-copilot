
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs"


def load_retriever():
    chunks = pd.read_csv(
        OUTPUT_DIR / "policy_chunks.csv",
        keep_default_na=False
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    chunk_vectors = vectorizer.fit_transform(
        chunks["text"]
    )

    return chunks, vectorizer, chunk_vectors


def retrieve(query, top_k=3):
    chunks, vectorizer, chunk_vectors = load_retriever()

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        chunk_vectors
    )[0]

    ranked_indices = similarities.argsort()[::-1][:top_k]

    results = []

    for index in ranked_indices:
        row = chunks.iloc[index]

        results.append({
            "document": row["document"],
            "chunk_id": row["chunk_id"],
            "score": round(float(similarities[index]), 4),
            "text": row["text"]
        })

    return results


def main():
    questions = [
        "When is delivery free?",
        "What is the refund policy?",
        "How is customer personal information protected?"
    ]

    print("PART D - POLICY RETRIEVAL")
    print("--------------------------------")

    for question in questions:
        print("\nQUESTION:", question)

        results = retrieve(question, top_k=3)

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result['document']} "
                f"| Score: {result['score']}"
            )

            print("   Chunk:", result["chunk_id"])


if __name__ == "__main__":
    main()
