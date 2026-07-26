# Research & Report Agent (LangGraph)

A scoped "deep research" agent: it plans sub-questions, researches them **in
parallel**, critiques its own coverage, loops back if there are gaps, and
synthesizes a final report.

Built to demonstrate LangGraph patterns that a linear LangChain chain can't
express cleanly: **dynamic parallel fan-out** (`Send`), **stateful
accumulation** across parallel branches, and a **reflection loop** with a
retry cap.

## Architecture

```
START
  |
planner            <- LLM breaks question into 3-5 sub-questions
  |
  |  (Send: one researcher node spawned PER sub-question, in parallel)
  v
researcher (xN)     <- each does a web search + LLM summarization
  |
  |  (all N branches join back into one state via a reducer)
  v
critique            <- LLM checks: does the evidence answer the
  |                     original question? any gaps?
  |
  +-- gaps found AND retries left --> back to planner (adds new
  |                                    sub-questions for the gaps)
  |
  +-- sufficient OR retries exhausted --> synthesizer --> END
```

## Why this shape (the two LangGraph ideas being demonstrated)

1. **Dynamic fan-out with `Send`** — the planner doesn't know in advance how
   many sub-questions there will be. `Send` lets a conditional edge spawn a
   variable number of parallel node executions at runtime, each with its own
   input, rather than hard-coding N researcher nodes.
2. **Reducer-based state merging** — each parallel researcher branch returns
   its own finding. Instead of the last one overwriting the others, the
   `findings` field uses `operator.add` as its reducer, so LangGraph merges
   all parallel results into one list automatically.

## Files

- `state.py` — the shared state schema (the "scoreboard")
- `tools.py` — web search tool (DuckDuckGo, no API key needed)
- `nodes.py` — planner / researcher / critique / synthesizer node functions
- `graph.py` — wires nodes into a `StateGraph`, including the `Send` fan-out
- `app.py` — Streamlit UI (matches the rest of your portfolio's deployment style)
- `main.py` — CLI entrypoint, for testing without Streamlit

## Setup

```bash
pip install -r requirements.txt
```

Set your Gemini API key (same provider you already used in the phishing
project's chatbot):

```bash
export GOOGLE_API_KEY="your-key-here"
```

## Run

CLI:
```bash
python main.py "Compare XGBoost vs LightGBM for tabular fraud detection"
```

Streamlit:
```bash
streamlit run app.py
```

## Extending it (good talking points for interviews)

- Swap `MemorySaver` for a persistent checkpointer (e.g. SQLite) to resume a
  run after a crash or inspect intermediate state — same idea as the
  checkpointing story from the self-eval-rag project, applied to a different
  control-flow pattern (planning/reflection instead of retrieval/retry).
- Add a `human_review` interrupt before the final report ships, using
  LangGraph's `interrupt()` — turns this into a human-in-the-loop system.
- Swap the DuckDuckGo tool for Tavily or a real search API for higher-quality
  results in production.
