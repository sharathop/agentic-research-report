"""
The shared state ("scoreboard") that flows through every node in the graph.

Most fields get OVERWRITTEN each time a node returns an update for them.
`findings` is different: because multiple `researcher` nodes run in
PARALLEL (one per sub-question), we don't want the last one to overwrite
the others. `operator.add` tells LangGraph "merge these lists together"
instead of "replace the old value" -- this is what makes fan-out/fan-in
work correctly.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Finding(TypedDict):
    sub_question: str
    answer: str
    sources: list[str]


class ResearchState(TypedDict):
    # Set once, at the start, never changes.
    question: str

    # Overwritten each time the planner runs (including re-runs after a
    # critique found gaps).
    sub_questions: list[str]

    # Accumulates across parallel researcher branches -- see docstring above.
    findings: Annotated[list[Finding], operator.add]

    # Overwritten by the critique node each pass.
    critique: str
    gaps_found: bool

    # Loop guard so a stubborn critique can't loop forever.
    retry_count: int
    max_retries: int

    # Set once, at the end, by the synthesizer.
    final_report: str


# The payload sent to each parallel researcher branch via `Send`.
# Only needs the one sub-question that branch is responsible for --
# it still has access to the rest of ResearchState when it runs.
class ResearcherInput(TypedDict):
    sub_question: str