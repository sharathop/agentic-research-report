# Research & Report Agent (LangGraph)

A scoped "deep research" agent: given a broad question, it plans sub-questions,
researches them **in parallel**, critiques its own coverage, loops back if
there are real gaps, and synthesizes a final report — honestly noting
whatever it couldn't fully answer instead of quietly papering over it.

Built to demonstrate the LangGraph patterns a linear LangChain chain can't
express cleanly: **dynamic parallel fan-out** (`Send`), **stateful
accumulation** across parallel branches, and a **reflection loop** with a
retry cap and honest failure reporting.

Live demo: https://agentic-research-report-fbzxpideaxaxj4ahksa8kd.streamlit.app

---

## The problem this solves

A single LLM call answering a broad question has three structural weaknesses:

1. **No decomposition** — one search or one pass can't properly cover a
   multi-part question ("compare X and Y" needs research on X, on Y, *and*
   on how they compare).
2. **No self-awareness of gaps** — it returns whatever it found, confidently,
   even if the answer is thin or one-sided.
3. **No verification loop** — there's no mechanism to notice a weak answer
   and go look again before committing to it.

This agent addresses all three: **plan → research in parallel → critique →
retry on real gaps → synthesize honestly.**

---

## Architecture

```
START
  |
planner            <- LLM breaks the question into 3-5 sub-questions
  |
  |  (Send: one researcher node spawned PER sub-question, in parallel)
  v
researcher (xN)     <- each does a web search (Tavily) + LLM summarization
  |
  |  (all N branches join back into one state via a reducer)
  v
critique            <- LLM checks: does the evidence answer the
  |                     original question? any real gaps?
  |
  +-- gaps found AND retries left --> back to planner (adds 2-3 NEW
  |                                    sub-questions targeting just the gap)
  |
  +-- sufficient OR retries exhausted --> synthesizer --> END
```

**Shared state** (flows through every node):
```python
question: str                                    # set once
sub_questions: list[str]                         # grows across retry passes
findings: Annotated[list[Finding], operator.add]  # merged from parallel branches
critique: str
gaps_found: bool
retry_count: int
max_retries: int
final_report: str
```

### Why this shape

- **Dynamic fan-out with `Send`** — the planner doesn't know in advance how
  many sub-questions there'll be. `Send` spawns a variable number of parallel
  `researcher` executions at runtime, each with its own input, instead of
  hardcoding N nodes.
- **Reducer-based state merging** — each parallel researcher branch returns
  its own finding. `findings` uses `operator.add` as its reducer, so
  LangGraph merges all parallel results into one list automatically instead
  of the last branch overwriting the others.
- **Targeted retries, not full restarts** — on a second pass, the planner
  generates 2-3 *new* sub-questions aimed specifically at the gap the
  critique found, and `dispatch_researchers` only re-runs the researcher
  node for those new ones — already-answered sub-questions aren't re-searched.
- **A retry cap, not unlimited looping** — `max_retries` (default 1) stops
  the reflection loop from running forever if the critique is never fully
  satisfied. When the cap is hit, the system says so explicitly rather than
  silently proceeding as if everything were resolved.

---

## Stack

- **LLM**: Groq (`llama-3.3-70b-versatile`) by default — free tier with much
  higher rate limits than Gemini's free tier, better for iterating during
  development. Falls back to Gemini (`gemini-2.5-flash-lite`) if
  `GROQ_API_KEY` isn't set, with a built-in rate limiter to stay under
  Gemini's stricter free-tier RPM.
- **Search**: Tavily — a search API purpose-built for LLM/agent use cases,
  returning cleaner, more structured results than a general-purpose search
  engine. Requires a free API key.
- **Orchestration**: LangGraph (`StateGraph`, `Send`, conditional edges,
  in-memory checkpointing).
- **UI**: Streamlit, streaming progress live as the graph executes.

No vector database — this is a pure live-web-research agent, not RAG. That's
intentional: it's meant to be a distinct project from a retrieval-over-your-own-documents
system, demonstrating a different skill (autonomous multi-step research over
external, current information) rather than overlapping with one.

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```bash
GROQ_API_KEY=your-groq-key       # https://console.groq.com
TAVILY_API_KEY=your-tavily-key   # https://tavily.com
# GOOGLE_API_KEY=your-gemini-key # optional fallback, only used if GROQ_API_KEY is unset
```

**Never commit `.env`** — add it to `.gitignore` before your first commit.

## Run

CLI:
```bash
python main.py "Compare RAG and fine-tuning for reducing LLM hallucination"
```

Streamlit:
```bash
streamlit run app.py
```

---

## Files

| File | Purpose |
|---|---|
| `state.py` | Shared state schema — the "scoreboard" every node reads/writes |
| `tools.py` | Tavily search + relevance filtering (see below) |
| `nodes.py` | Planner / researcher / critique / synthesizer node functions |
| `graph.py` | Wires nodes into a `StateGraph`, including the `Send` fan-out |
| `app.py` | Streamlit UI, streaming graph progress live |
| `main.py` | CLI entrypoint, no UI needed |

---

## What broke during testing, and what I learned fixing it

Building this surfaced real, non-obvious bugs — documenting them here
because the debugging process is arguably more instructive than the happy
path.

**1. Critique label contradicted its own text.** The UI showed "✅ Coverage
sufficient" right next to critique text explaining exactly why coverage
*wasn't* sufficient. Root cause: the retry-cap logic force-set `gaps_found`
to `False` once retries ran out, but the label rendering didn't distinguish
"genuinely sufficient" from "stopped because we hit the retry limit." Fixed
by making the retry-cap override explicit in the critique text itself
("Retry limit reached — proceeding despite these gaps") instead of leaving
a bare, misleading checkmark.

**2. Naive keyword-overlap relevance filtering was fragile in both
directions.** Early on, Tavily occasionally returned results for the
*literal word* in a question ("What is the definition of RAG?" → search
results about the word "what") — noise fed straight into the summarizer,
which dutifully reported "the results don't address the topic," an honest
but useless outcome. Adding a keyword-overlap check (require N shared
non-stopword keywords between the query and each result) fixed that — but
requiring 2 shared keywords was then too strict for long questions,
occasionally returning zero results after filtering ("No search results
found") for genuinely researchable sub-questions. Loosening to 1 shared
keyword fixed that, but reintroduced the original problem in a new form:
generic question-template words that aren't true stopwords ("potential,"
"key," "future," "approach") let clearly off-topic results slip back in.

The real lesson: **pure keyword-overlap matching is structurally fragile**
for this use case — it can't distinguish topical relevance from incidental
word overlap. Expanding the stopword list to cover common
LLM-generated-question filler words closed the gap for questions tested so
far, but this is a patch, not a fundamental fix. A more robust version
would use an LLM call to judge relevance directly, or anchor relevance
checks to the *original* research question's core terms rather than each
sub-question's incidental phrasing.

**3. One search-tool failure shouldn't crash the whole run.** With 5-8
parallel researcher branches firing per run, a single transient Tavily
error would otherwise take down the entire graph execution, discarding
every other branch's completed work. Each researcher branch now catches
its own search failure, records that one sub-question as failed, and lets
the rest of the run complete normally.

**4. Findings the system already knew were missing got silently dropped
from the final report.** The critique step would correctly identify a gap,
but that finding never reached the synthesizer — which only received the
raw research findings, not the critique's verdict. The report would read
as if nothing were missing. Fixed by passing the critique text into the
synthesizer's prompt, with an explicit instruction to either address the
gap using available findings or state plainly, in a short "Limitations"
section, that it couldn't be closed — rather than staying silent about a
gap the system already knew existed.

---

## Known limitations

- Keyword-based relevance filtering (see above) is a reasonable heuristic,
  not a robust solution — it can still occasionally misclassify results
  when a generic word isn't yet in the stopword list.
- No fact-checking across sources — if two sources disagree, the
  synthesizer doesn't currently flag the contradiction explicitly.
- Retry cap defaults to 1 — sufficient for most questions tested, but a
  genuinely broad question may need 2 to fully close gaps (at the cost of
  more API calls and latency).

## Extending it (good talking points)

- Swap `MemorySaver` for a persistent checkpointer (SQLite/Postgres) to
  resume a run after a crash or inspect intermediate state.
- Add a `human_review` interrupt before the report ships, using
  LangGraph's `interrupt()` — turns this into a human-in-the-loop system.
- Replace keyword-overlap relevance filtering with an LLM-judged relevance
  check per search result — more reliable, at the cost of one extra LLM
  call per result.
- Anchor relevance checks to the original research question's core terms
  (stable across a run) rather than each sub-question's incidental
  phrasing (which varies and can introduce generic filler words).