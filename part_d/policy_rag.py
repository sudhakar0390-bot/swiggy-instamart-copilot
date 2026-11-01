
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm
from retrieve_policies import retrieve


def answer_policy_question(question):
    # Step 1: Retrieve the top 3 policy chunks
    results = retrieve(question, top_k=3)

    # Step 2: Reject questions without relevant matches
    relevant = [
        result for result in results
        if result["score"] > 0
    ]

    if not relevant:
        return {
            "answer": "I could not find a relevant policy. Please contact support.",
            "sources": [],
            "tokens": 0
        }

    # Step 3: Prepare the retrieved policy context
    context = "\n\n".join(
        f"Source: {result['document']}\n"
        f"Content: {result['text']}"
        for result in relevant
    )

    # Step 4: Prepare the messages for mock_llm
    
    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. "
                "Answer the user's question using only the "
                "provided policy context. "
                "Do not invent policy rules. "
                "If the answer is not supported, say so."
            )
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{question}"
            )
        }
    ]


    # Step 5: Call the offline mock LLM
    response = None

    for attempt in range(3):
        try:
            response = mock_llm.chat(messages)
            break
        except mock_llm.TransientLLMError:
            if attempt == 2:
                return {
                    "answer": "The policy assistant is temporarily unavailable.",
                    "sources": [],
                    "tokens": 0
                }

            time.sleep(0.1 * (2 ** attempt))

    # Step 6: Extract the response
    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [
            result["document"] for result in relevant
        ],
        "tokens": response.usage.total_tokens
    }


def main():
    mock_llm.reset()

    questions = [
        "When is delivery free?",
        "What is the refund policy?",
        "How is customer personal information protected?"
    ]

    print("PART D - POLICY RAG ASSISTANT")
    print("--------------------------------")

    for question in questions:
        result = answer_policy_question(question)

        print("\nQUESTION:", question)
        print("ANSWER:", result["answer"])
        print("SOURCES:", ", ".join(result["sources"]))
        print("TOKENS:", result["tokens"])


if __name__ == "__main__":
    main()
