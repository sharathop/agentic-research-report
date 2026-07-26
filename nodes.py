"""
Each function here is one node in the graph. A node's job is simple and
uniform: read whatever fields it needs from `state`, do its work, and
return a dict of ONLY the fields it wants to update. LangGraph merges that
dict into the running state (using each field's reducer -- see state.py).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from state import ResearcherInput, ResearchState
from tools import web_search

load_dotenv()

# Built lazily (see _get_gemini_rate_limiter below) so neither this object
# nor the langchain_google_genai import it supports get created at all on
# a Groq-only setup.
_gemini_rate_limiter = None


def _get_gemini_rate_limiter():
    """
    Smooths out request bursts -- without this, the parallel researcher
    fan-out fires several LLM calls at nearly the same instant, which can
    trip the per-minute limit even before you hit the daily cap. One shared
    limiter across all calls keeps requests spaced out no matter how many
    nodes are running concurrently. Only needed for Gemini -- Groq's free
    tier limits are high enough that this isn't a practical concern there.
    Built once and cached (module-level singleton) so every Gemini call
    shares the same limiter instead of getting its own independent one.
    """
    global _gemini_rate_limiter
    if _gemini_rate_limiter is None:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        _gemini_rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.12,  # ~1 request every 8-9s, safely under 10 RPM
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    return _gemini_rate_limiter


def _llm(temperature: float = 0.2):
    """
    Uses Groq if GROQ_API_KEY is set (much higher free-tier limits, good
    for iterating during development), otherwise falls back to Gemini.
    Both return a LangChain chat model with the same .invoke() interface,
    so nothing else in this file needs to know which one is active.
    """
    if os.environ.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=os.environ.get("GROQ_API_KEY"),
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=temperature,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        rate_limiter=_get_gemini_rate_limiter(),
    )


# ---------------------------------------------------------------------------
# 1. Planner -- breaks the question into sub-questions.
#    Also the node that runs again after a critique finds gaps, this time
#    asked to produce sub-questions that specifically fill those gaps.
# ---------------------------------------------------------------------------

class SubQuestions(BaseModel):
    sub_questions: list[str] = Field(
        description="3 to 5 focused, independently-researchable sub-questions"
    )


def planner_node(state: ResearchState) -> dict:
    llm = _llm().with_structured_output(SubQuestions)

    if state.get("critique"):
        # Second-pass planning: fill the gaps the critique identified,
        # don't just repeat the original breakdown.
        prompt = ChatPromptTemplate.from_template(
            "Original research question: {question}\n\n"
            "A critique found these gaps in the research so far:\n"
            "{critique}\n\n"
            "Produce 2-3 NEW, focused sub-questions that specifically "
            "address these gaps. Do not repeat sub-questions already "
            "covered: {existing}"
        )
        result: SubQuestions = (prompt | llm).invoke(
            {
                "question": state["question"],
                "critique": state["critique"],
                "existing": state["sub_questions"],
            }
        )
        # Append new sub-questions rather than replacing, so the researcher
        # fan-out below covers the gap questions this round, and the state
        # field keeps reflecting every sub-question asked so far.
        return {"sub_questions": state["sub_questions"] + result.sub_questions}

    prompt = ChatPromptTemplate.from_template(
        "Break this research question into 3-5 focused, independently "
        "researchable sub-questions. Question: {question}"
    )
    result: SubQuestions = (prompt | llm).invoke({"question": state["question"]})
    return {"sub_questions": result.sub_questions, "retry_count": 0}


# ---------------------------------------------------------------------------
# 2. Researcher -- runs once per sub-question, IN PARALLEL (via Send).
#    Note the input type: `ResearcherInput`, not the full `ResearchState`.
#    LangGraph's Send lets a node receive a narrower payload than the full
#    state; it still returns updates that get merged into the full state.
# ---------------------------------------------------------------------------

def researcher_node(payload: ResearcherInput) -> dict:
    sub_question = payload["sub_question"]

    try:
        results = web_search(sub_question)
    except RuntimeError as exc:
        # web_search() raises (rather than silently degrading) on a
        # missing/bad Tavily key, missing package, or a failed API call --
        # see tools.py. That's the right behavior at the search-tool level,
        # but one sub-question's search failing (e.g. a transient rate
        # limit) shouldn't take down the whole parallel research run.
        # Record it as a failed finding and let the other branches finish;
        # the critique step will see this gap like any other and can
        # decide whether it's worth a retry.
        print(f"[researcher] search failed for {sub_question!r}: {exc}")
        return {
            "findings": [
                {
                    "sub_question": sub_question,
                    "answer": f"Search failed for this sub-question: {exc}",
                    "sources": [],
                }
            ]
        }

    if not results:
        return {
            "findings": [
                {
                    "sub_question": sub_question,
                    "answer": "No search results found.",
                    "sources": [],
                }
            ]
        }

    context = "\n\n".join(
        f"[{r['title']}]({r['url']})\n{r['snippet']}" for r in results
    )
    prompt = ChatPromptTemplate.from_template(
        "Using ONLY the search results below, answer the sub-question in "
        "2-4 sentences. Be concrete; note if the results are inconclusive.\n\n"
        "Sub-question: {sub_question}\n\nSearch results:\n{context}"
    )
    answer = (prompt | _llm()).invoke(
        {"sub_question": sub_question, "context": context}
    ).content

    return {
        "findings": [
            {
                "sub_question": sub_question,
                "answer": answer,
                "sources": [r["url"] for r in results],
            }
        ]
    }


# ---------------------------------------------------------------------------
# 3. Critique -- decides whether the findings actually answer the original
#    question, or whether there are gaps worth another research pass.
# ---------------------------------------------------------------------------

class CritiqueResult(BaseModel):
    gaps_found: bool
    critique: str = Field(
        description="If gaps_found, describe specifically what's missing. "
        "If not, briefly say why the coverage is sufficient."
    )


def critique_node(state: ResearchState) -> dict:
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)

    findings_text = "\n\n".join(
        f"Q: {f['sub_question']}\nA: {f['answer']}" for f in state["findings"]
    )
    llm = _llm().with_structured_output(CritiqueResult)
    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\n\n"
        "Findings gathered so far:\n{findings}\n\n"
        "Do these findings, taken together, sufficiently answer the "
        "original question? Identify any real gaps, not nitpicks."
    )
    result: CritiqueResult = (prompt | llm).invoke(
        {"question": state["question"], "findings": findings_text}
    )

    # Loop guard: if we've hit max_retries, force gaps_found to False so
    # the graph moves on to synthesis instead of looping forever.
    hit_retry_limit = result.gaps_found and retry_count >= max_retries
    gaps_found = result.gaps_found and not hit_retry_limit

    critique_text = result.critique
    if hit_retry_limit:
        # The LLM still found gaps, but the loop guard is overriding
        # gaps_found to False -- say so explicitly instead of leaving the
        # original "insufficient" text next to a now-misleading checkmark.
        critique_text = (
            f"{result.critique}\n\n"
            f"(Retry limit reached ({max_retries}) — proceeding to synthesis "
            f"despite these gaps.)"
        )

    return {
        "critique": critique_text,
        "gaps_found": gaps_found,
        "retry_count": retry_count + 1,
    }


# ---------------------------------------------------------------------------
# 4. Synthesizer -- combines all findings into one final report.
# ---------------------------------------------------------------------------

def synthesizer_node(state: ResearchState) -> dict:
    findings_text = "\n\n".join(
        f"### {f['sub_question']}\n{f['answer']}\n"
        f"Sources: {', '.join(f['sources']) or 'none'}"
        for f in state["findings"]
    )

    critique = state.get("critique", "")
    if critique:
        # Step A's note, now actually handed to Step B instead of thrown
        # away. If gaps were found but we're moving on anyway (retry limit
        # hit), the report should say so honestly -- e.g. a short
        # "Limitations" note -- rather than reading as if nothing were
        # missing.
        prompt = ChatPromptTemplate.from_template(
            "Write a clear, well-organized report answering this research "
            "question, using the findings below. Use headings. Cite "
            "sources inline where relevant.\n\n"
            "A critique of these findings noted the following gaps:\n"
            "{critique}\n\n"
            "Where possible, address these gaps directly using the "
            "findings available. For any gap that genuinely can't be "
            "closed with what's here, say so explicitly in a short "
            "'Limitations' section at the end, rather than ignoring it.\n\n"
            "Question: {question}\n\nFindings:\n{findings}"
        )
        report = (prompt | _llm()).invoke(
            {
                "question": state["question"],
                "findings": findings_text,
                "critique": critique,
            }
        ).content
        return {"final_report": report}

    prompt = ChatPromptTemplate.from_template(
        "Write a clear, well-organized report answering this research "
        "question, using the findings below. Use headings. Cite sources "
        "inline where relevant.\n\n"
        "Question: {question}\n\nFindings:\n{findings}"
    )
    report = (prompt | _llm()).invoke(
        {"question": state["question"], "findings": findings_text}
    ).content
    return {"final_report": report}