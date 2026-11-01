
from pathlib import Path
import sys
import json
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm
from checkpoint import save_checkpoint, load_checkpoint, clear_checkpoint
from refund_tools import (
    get_order,
    search_policy,
    calculate_refund,
    issue_refund,
    escalate_to_human
)

MAX_ITERATIONS = 6
TOKEN_LIMIT = 6000
TIMEOUT_SECONDS = 10
LOG_PATH = ROOT / "outputs" / "agent_runs.jsonl"
TOOL_FUNCTIONS = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


def save_agent_log(user_request, final_response, trace, total_tokens):
    """Save one completed agent run to a JSONL file."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "user_request": user_request,
        "final_response": final_response,
        "total_tokens": total_tokens,
        "tool_trace": trace
    }

    with open(LOG_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

def run_agent(
    user_request,
    max_iterations=MAX_ITERATIONS,
    token_limit=TOKEN_LIMIT,
    timeout_seconds=TIMEOUT_SECONDS,
    resume=False
):
    messages = [
        {
            "role": "system",
            "content": (
                "AGENT. You are a refund support assistant. "
                "Use the available tools to check orders, "
                "retrieve policies and process eligible refunds. "
                "Do not issue refunds without verifying eligibility."
            )
        },
        {
            "role": "user",
            "content": user_request
        }
    ]

    start_time = time.monotonic()
    total_tokens = 0
    trace = []
    start_iteration = 0

    if resume:
        checkpoint = load_checkpoint()

        if checkpoint is None:
            raise ValueError("No checkpoint available to resume.")

        saved_request = checkpoint["messages"][1]["content"]

        if saved_request != user_request:
            raise ValueError("Checkpoint belongs to a different request.")

        messages = checkpoint["messages"]
        start_iteration = checkpoint["iteration"]
        total_tokens = checkpoint["total_tokens"]
        trace = checkpoint["trace"]

        print("Resuming from iteration:", start_iteration)

    else:
        clear_checkpoint()

    print("\nCUSTOMER REQUEST:", user_request)

    for iteration in range(start_iteration, max_iterations):
        if time.monotonic() - start_time >= timeout_seconds:
            return "FALLBACK_TIMEOUT"

        if total_tokens >= token_limit:
            return "FALLBACK_TOKEN_LIMIT"

        response = mock_llm.chat(
            messages,
            tools=list(TOOL_FUNCTIONS.keys())
        )

        total_tokens += response.usage.total_tokens

        if total_tokens > token_limit:
            return "FALLBACK_TOKEN_LIMIT"

        message = response.choices[0].message

        
        if response.choices[0].finish_reason != "tool_calls":
            final_response = message.content

            save_agent_log(
                user_request,
                final_response,
                trace,
                total_tokens
            )
            clear_checkpoint()
            return final_response
    
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in message.tool_calls
            ]
        })

        for tool_call in message.tool_calls:
            tool_name = tool_call.name
            arguments = tool_call.arguments

            print(
                f"Iteration {iteration + 1}:",
                tool_name,
                arguments
            )

            if tool_name not in TOOL_FUNCTIONS:
                result = {"error": "Unknown tool"}
            else:
                try:
                    result = TOOL_FUNCTIONS[tool_name](**arguments)
                except Exception as error:
                    result = {"error": str(error)}

            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result)
            })

            trace.append({
                "iteration": iteration + 1,
                "tool": tool_name,
                "arguments": arguments,
                "result": result
            })

            print("Tool result:", result)

        save_checkpoint(
            messages,
            iteration + 1,
            total_tokens,
            trace
        )


    return "FALLBACK_MAX_ITERATIONS"




def main():
    mock_llm.reset()

    result = run_agent("Refund order ORD-108250")

    print("\nFINAL AGENT RESPONSE:", result)
    print("Log file:", LOG_PATH)


if __name__ == "__main__":
    main()

