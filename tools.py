"""
Search tool used by the researcher node.

Uses Tavily only -- purpose-built for LLM/agent search, needs a free API
key from https://tavily.com (set TAVILY_API_KEY in your .env).

`web_search()` raises a clear error immediately if the key is missing or
the package isn't installed, instead of silently degrading to a weaker
fallback provider -- better to fail loudly at startup than to quietly get
worse search results without knowing why.
"""

from __future__ import annotations

import os
import re

# Small, deliberately generic list -- just enough to ignore question
# scaffolding ("what", "does", "the"...) so the overlap check below is
# comparing the words that actually carry meaning, not every word.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "between", "by", "do", "does",
    "for", "from", "how", "in", "is", "it", "its", "of", "on", "or",
    "specific", "that", "the", "this", "to", "what", "with",
    # Common in LLM-generated sub-question TEMPLATES ("What are the
    # potential X and Y..."), not actual topic content -- without these,
    # a page that happens to be about the generic word itself (e.g. a
    # dictionary/physics page for "potential") can slip past the
    # relevance check just by sharing that one filler word.
    "potential", "key", "effective", "effectiveness", "way", "ways",
    "future", "role", "approach", "approaches",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _is_relevant(query_keywords: set[str], result: dict) -> bool:
    """
    True if the result's title/snippet shares at least one real keyword
    with the query. Just 1 shared keyword is enough -- for long,
    multi-part questions, a genuinely relevant result often only overlaps
    on a single strong term (e.g. "limitations") because it's phrased
    differently than the question, not because it's off-topic. Requiring
    2+ was filtering those out entirely, sometimes leaving zero results.
    A single truly unrelated result (e.g. a dictionary page for "what")
    is still caught, since it shares 0 real keywords once stopwords are
    removed -- 1 is the right bar, not 2.
    """
    if not query_keywords:
        return True  # nothing meaningful to check against, don't filter
    result_text = f"{result.get('title', '')} {result.get('snippet', '')}"
    overlap = query_keywords & _keywords(result_text)
    return len(overlap) >= 1


def _filter_relevant(query: str, results: list[dict]) -> list[dict]:
    query_keywords = _keywords(query)
    filtered = [r for r in results if _is_relevant(query_keywords, r)]
    dropped = len(results) - len(filtered)
    if dropped:
        print(f"[web_search] dropped {dropped} irrelevant result(s) for {query!r}")
    return filtered


def _tavily_search(query: str, max_results: int) -> list[dict]:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Add it to your .env file -- get a "
            "free key at https://tavily.com."
        )

    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise RuntimeError(
            "tavily-python isn't installed -- run: pip install tavily-python"
        ) from exc

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=max_results)
    except Exception as exc:  # noqa: BLE001 -- surface it, don't hide it
        raise RuntimeError(f"Tavily search failed for {query!r}: {exc}") from exc

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]


def web_search(query: str, max_results: int = 4) -> list[dict]:
    """
    Returns a list of {"title": ..., "url": ..., "snippet": ...} dicts
    from Tavily, filtered to drop anything that shares no real keyword
    with the query -- e.g. a dictionary definition of "what" that
    happened to rank for a question starting with "What is...".

    Retries the raw search once if it comes back with zero raw results --
    covers transient Tavily flakiness rather than immediately reporting
    "No search results found" for what might just be a one-off hiccup.
    Does NOT retry if results came back but all got filtered out by
    relevance -- that's a real "nothing relevant" outcome, not a fluke.
    """
    results = _tavily_search(query, max_results)
    if not results:
        print(f"[web_search] zero raw results for {query!r}, retrying once")
        results = _tavily_search(query, max_results)
    return _filter_relevant(query, results)