"""CLI entrypoint. Usage: python main.py "your research question"""

from __future__ import annotations

import sys
import uuid

from graph import build_graph


def run(question: str) -> str:
    app = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "question": question,
        "sub_questions": [],
        "findings": [],
        "critique": "",
        "gaps_found": False,
        "retry_count": 0,
        "max_retries": 1,
        "final_report": "",
    }

    final_state = None
    for step in app.stream(initial_state, config=config, stream_mode="values"):
        final_state = step
        if step.get("sub_questions"):
            print(f"[planner] sub-questions: {step['sub_questions']}")
        if step.get("critique"):
            print(f"[critique] gaps_found={step['gaps_found']}: {step['critique']}")

    return final_state["final_report"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py "your research question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    report = run(question)
    print("\n=== FINAL REPORT ===\n")
    print(report)
