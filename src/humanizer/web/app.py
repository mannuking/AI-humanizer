"""The Streamlit interface.

Launched by ``uv run humanize``. The layout follows the shape of the
pipeline: put text in on the left, see the analysis and the rewrite on the
right, and keep the score visible so the effect of a change is obvious.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make the package importable when run as `streamlit run src/humanizer/web/app.py`
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from humanizer.analyze import analyze  # noqa: E402
from humanizer.config import load_config  # noqa: E402
from humanizer.llm import MiniMaxError, build_client  # noqa: E402
from humanizer.pipeline import Humanizer  # noqa: E402
from humanizer.profiles import PROFILES  # noqa: E402

st.set_page_config(
    page_title="Humanizer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG = load_config()


# ---------------------------------------------------------------- helpers
def score_color(score: float) -> str:
    if score < 0.15:
        return "#10b981"
    if score < 0.35:
        return "#84cc16"
    if score < 0.60:
        return "#f59e0b"
    return "#ef4444"


def score_bar(score: float, label: str) -> None:
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:0.82rem;color:#64748b;margin-bottom:2px'>"
        f"<span>{label}</span><span>{score:.3f}</span></div>",
        unsafe_allow_html=True,
    )
    st.progress(min(score, 1.0))


def render_analysis(report, container=None) -> None:
    """Render the analysis panel."""
    target = container if container is not None else st
    stats = report.stats

    col_a, col_b, col_c = target.columns(3)
    col_a.metric("AI score", f"{report.overall_score:.3f}")
    col_b.metric("Patterns found", len(report.findings))
    col_c.metric("Words", stats.word_count)

    with target.expander("Breakdown", expanded=False):
        score_bar(report.pattern_score, "Pattern density")
        score_bar(report.style_score, "Style uniformity")
        st.markdown("**Shape**")
        st.json(
            {
                "burstiness": stats.burstiness,
                "type_token_ratio": stats.type_token_ratio,
                "hapax_ratio": stats.hapax_ratio,
                "mean_sentence_length": stats.mean_sentence_length,
                "sentence_length_stdev": stats.stdev_sentence_length,
                "punctuation_entropy": stats.punctuation_entropy,
                "reading_ease": stats.reading_ease,
                "sentences": stats.sentence_count,
                "paragraphs": stats.paragraph_count,
                "em_dash_rate": stats.em_dash_rate,
                "comma_rate": stats.comma_rate,
            }
        )

    if report.findings:
        with target.expander(f"What to fix ({len(report.findings)})", expanded=True):
            rows = [
                {
                    "line": f.detector and f.line,
                    "pattern": f.detector,
                    "match": f.match[:60],
                    "suggestion": f.suggestion,
                }
                for f in report.top_findings(40)
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ✍️ Humanizer")
    st.caption(
        "Rewrites AI-assisted text so it reads the way a person actually "
        "writes, and tells you what it changed."
    )

    st.divider()

    if CONFIG.has_key:
        st.success(f"Connected to `{CONFIG.model}`")
    else:
        st.error("MINIMAX_API_KEY is not set")
        st.caption(
            "Add `MINIMAX_API_KEY=...` to the project `.env` file, then "
            "restart the app."
        )

    st.markdown("#### Target style")
    profile_label = st.selectbox(
        "Profile",
        options=list(PROFILES.keys()),
        format_func=lambda k: f"{PROFILES[k].label} — {PROFILES[k].description}",
        index=1,
    )

    st.markdown("#### Rewrite settings")
    max_passes = st.slider("Maximum passes", 1, 5, CONFIG.max_passes)
    preserve_length = st.checkbox(
        "Preserve length", value=CONFIG.preserve_word_count,
        help="Aim for the same word count. The incumbent tools inflate length "
             "by 35% on average; keeping the length honest is one of the "
             "things this tool does better.",
    )
    verify_meaning = st.checkbox(
        "Verify meaning", value=True,
        help="Run a second check that the rewrite says the same thing as the "
             "original, and revert if it does not.",
    )
    tone = st.text_area(
        "Extra guidance (optional)",
        placeholder="e.g. keep the first person voice, keep the technical terms",
        height=80,
    )

    st.divider()
    if st.button("How it works", use_container_width=True):
        st.session_state["show_help"] = not st.session_state.get("show_help", False)

    if st.session_state.get("show_help"):
        st.markdown(
            """
**The pipeline**

1. **Analyse** — 30 detectors look for the tells of machine-written prose,
   plus a statistical pass over sentence rhythm and vocabulary.
2. **Protect** — numbers, dates, citations, code, URLs and tables are
   hidden behind placeholders so the model cannot damage them.
3. **Rewrite** — the passage is rewritten in blocks, with the detected
   tells named in the instructions.
4. **Restore** — protected values go back, then they are *verified*.
5. **Verify** — a second call checks that the meaning survived.
6. **Re-score** — if the score is still high, another pass runs, now
   knowing what the last pass missed.

**What this does not do**

It does not insert typos, shuffle sentences or mangle spelling. Those
tricks damage the text and are still caught. It does not invent facts.
            """
        )


# ---------------------------------------------------------------- main
st.markdown("## Rewrite a passage")

input_col, output_col = st.columns([1, 1], gap="large")

with input_col:
    st.markdown("#### Source text")
    source = st.text_area(
        "Paste the text to rewrite",
        height=420,
        key="source",
        placeholder="Paste a paragraph, a section, or a whole document.",
    )
    words = len(source.split()) if source else 0
    st.caption(f"{words} words")

    analyse_now = st.button("Analyse only (no API calls)", use_container_width=True)

with output_col:
    st.markdown("#### Result")
    output_area = st.empty()

if analyse_now:
    if not source.strip():
        st.warning("Paste some text first.")
    else:
        report = analyze(source)
        with output_area.container():
            render_analysis(report)

if st.button("✍️ Rewrite", type="primary", use_container_width=True):
    if not source.strip():
        st.warning("Paste some text first.")
    elif not CONFIG.has_key:
        st.error("Set MINIMAX_API_KEY in the .env file first.")
    else:
        progress = st.progress(0.0, text="Starting...")

        def on_progress(message: str) -> None:
            progress.text(message)

        try:
            client = build_client(CONFIG)
        except MiniMaxError as exc:
            st.error(str(exc))
        else:
            engine = Humanizer(client, CONFIG, on_progress=on_progress)
            try:
                result = engine.rewrite(
                    source,
                    profile_key=profile_label,
                    tone_instructions=tone,
                    max_passes=max_passes,
                    preserve_word_count=preserve_length,
                    verify_meaning=verify_meaning,
                )
            except MiniMaxError as exc:
                progress.empty()
                st.error(f"The API call failed: {exc}")
            else:
                progress.empty()
                with output_area.container():
                    head_a, head_b, head_c, head_d = st.columns(4)
                    head_a.metric("Before", f"{result.original_report.overall_score:.3f}")
                    head_b.metric("After", f"{result.final_report.overall_score:.3f}")
                    head_c.metric("Improvement", f"{result.improvement:+.3f}")
                    head_d.metric(
                        "Length",
                        f"{result.word_count_after / max(result.word_count_before, 1):.2f}x",
                    )

                    st.text_area(
                        "Rewritten text",
                        value=result.final_text,
                        height=420,
                        key="result_text",
                        disabled=True,
                    )

                    col_copy, col_dl = st.columns(2)
                    with col_copy:
                        st.code(result.final_text[:2000] + ("..." if len(result.final_text) > 2000 else ""),
                                language=None)
                    with col_dl:
                        st.download_button(
                            "Download .md",
                            data=result.final_text,
                            file_name="humanized.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )

                    if result.notes:
                        with st.expander("Notes"):
                            for note in result.notes:
                                st.warning(note)

                    with st.expander("Analysis of the result"):
                        render_analysis(result.final_report)


# ---------------------------------------------------------------- footer
st.divider()
st.caption(
    "This tool rewrites prose and reports what it found. It cannot determine "
    "who wrote a text, and a score is a style estimate, not proof of "
    "authorship. Follow your institution's policy on AI-assisted writing."
)

if os.getenv("HUMANIZER_DEBUG"):
    st.write("debug: config", CONFIG)
