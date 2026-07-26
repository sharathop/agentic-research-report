"""Streamlit UI. Run with: streamlit run app.py"""

from __future__ import annotations

import uuid

import streamlit as st

from graph import build_graph

st.set_page_config(page_title="Research & Report Agent")
st.title(" Research & Report Agent")
st.caption(
    "Plans sub-questions, researches them in parallel, critiques its own "
    "coverage, and synthesizes a report — built with LangGraph."
)

question = st.text_input(
    "Research question",
    placeholder="Compare XGBoost vs LightGBM for tabular fraud detection",
)

if st.button("Run", type="primary") and question:
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

    planner_box = st.empty()
    critique_box = st.empty()
    findings_slot = st.empty()

    with st.spinner("Running the graph..."):
        for step in app.stream(initial_state, config=config, stream_mode="values"):
            if step.get("sub_questions"):
                planner_box.info(f"**Sub-questions:** {step['sub_questions']}")
            if step.get("findings"):
                with findings_slot.container():
                    with st.expander("Findings gathered", expanded=False):
                        for f in step["findings"]:
                            st.markdown(f"**{f['sub_question']}**\n\n{f['answer']}")
            if step.get("critique"):
                status = "🔁 Gaps found — researching more" if step["gaps_found"] else "✅ Coverage sufficient"
                critique_box.warning(f"**Critique:** {status}\n\n{step['critique']}")

    st.markdown("## Final Report")
    st.markdown(step["final_report"])