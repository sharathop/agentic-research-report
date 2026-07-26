"""
Wires the node functions from nodes.py into an actual LangGraph StateGraph.

The one non-obvious part is `dispatch_researchers`: instead of a normal
conditional edge that picks ONE next node, it returns a LIST of `Send`
objects. Each `Send("researcher", {...})` spawns a separate parallel
execution of the researcher node with its own input. The number of
researchers spawned is decided at RUNTIME based on how many sub-questions
the planner produced -- that's the "dynamic fan-out" this project is
built to demonstrate.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from nodes import critique_node, planner_node, researcher_node, synthesizer_node
from state import ResearchState


def dispatch_researchers(state: ResearchState) -> list[Send]:
    """
    Conditional edge run right after the planner. Spawns one parallel
    `researcher` node per sub-question that doesn't already have a
    finding -- so on the second pass (after a critique found gaps), only
    the NEW sub-questions get dispatched, not ones already researched.
    """
    already_covered = {f["sub_question"] for f in state.get("findings", [])}
    pending = [q for q in state["sub_questions"] if q not in already_covered]
    return [Send("researcher", {"sub_question": q}) for q in pending]


def route_after_critique(state: ResearchState) -> str:
    return "planner" if state["gaps_found"] else "synthesizer"


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critique", critique_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")

    # Dynamic fan-out: planner -> N parallel researcher instances.
    graph.add_conditional_edges("planner", dispatch_researchers, ["researcher"])

    # Fan-in happens implicitly: LangGraph waits for all spawned `researcher`
    # branches to finish (merging `findings` via its reducer) before
    # continuing to the next node all of them point to.
    graph.add_edge("researcher", "critique")

    # Reflection loop: go back to planner for another pass, or move on.
    graph.add_conditional_edges(
        "critique", route_after_critique, ["planner", "synthesizer"]
    )

    graph.add_edge("synthesizer", END)

    # In-memory checkpointing so a run's state can be inspected/resumed.
    # Swap for a SQLite/Postgres checkpointer for real persistence.
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
