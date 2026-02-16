"""
Parallel LLM Parsing test: run Gemini 4o nano, Claude 4.5 Haiku, and Gemini 2.5 Flash
on all CVs in the ground truth database (parsed text → structured JSON).
Results are stored in SQLite (ground_truth_database/llm_parsing_test_results.db) and persist after closing the app.
Based on llm_parsing_benchmark.py; models run in parallel per CV (one thread per model).
Gemini 4o nano uses Replicate's Gemini 2.5 Flash (2.0 Flash not available on Replicate).
"""
import json
import os
import sqlite3
import time
from datetime import datetime
from threading import Thread
from typing import Any, Optional

import streamlit as st

# Import shared paths and metadata (project root + pages for sibling modules)
import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_this_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
import ocr_common as ocr

# Reuse run logic and accuracy from LLM parsing benchmark (lives in pages/)
from llm_parsing_benchmark import (
    _accuracy_vs_ground_truth,
    _run_one_model,
    get_replicate_key,
)

# ---------------------------------------------------------------------------
# Models: Gemini 4o nano, Claude, Gemini 2.5 Flash only (no GPT-5 mini).
# Replicate has no Gemini 2.0 Flash; we use Gemini 2.5 Flash for the "4o nano" slot.
# ---------------------------------------------------------------------------
PARSING_MODELS = {
    "replicate/gemini-4o-nano": {
        "label": "Gemini 4o nano",
        "provider": "replicate",
        "replicate_id": "google/gemini-2.5-flash",
        "cost_estimate_per_run": 0.001,
        "structured_output": False,
    },
    "replicate/claude-4.5-haiku": {
        "label": "Claude 4.5 Haiku",
        "provider": "replicate",
        "replicate_id": "anthropic/claude-4.5-haiku",
        "cost_estimate_per_run": 0.002,
        "structured_output": False,
    },
    "replicate/gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash",
        "provider": "replicate",
        "replicate_id": "google/gemini-2.5-flash",
        "cost_estimate_per_run": 0.001,
        "structured_output": False,
    },
}

# Inject these into llm_parsing_benchmark's MODELS so _run_one_model works
import llm_parsing_benchmark as _bench
_bench.MODELS.update(PARSING_MODELS)

# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(ocr.DB_DIR, "llm_parsing_test_results.db")


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_parsing_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cv_filename TEXT NOT NULL,
                model_name TEXT NOT NULL,
                language TEXT,
                layout_type TEXT,
                accuracy_pct REAL,
                time_seconds REAL,
                cost_usd REAL,
                json_valid INTEGER,
                schema_valid INTEGER,
                error TEXT,
                run_at TEXT
            )
        """)
        conn.commit()


def insert_result(
    cv_filename: str,
    model_name: str,
    language: str,
    layout_type: str,
    accuracy_pct: Optional[float],
    time_seconds: float,
    cost_usd: float,
    json_valid: bool,
    schema_valid: bool,
    error: Optional[str],
):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO llm_parsing_test_results
            (cv_filename, model_name, language, layout_type, accuracy_pct,
             time_seconds, cost_usd, json_valid, schema_valid, error, run_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cv_filename, model_name, language, layout_type, accuracy_pct,
            time_seconds, cost_usd, 1 if json_valid else 0, 1 if schema_valid else 0,
            error, datetime.utcnow().isoformat() + "Z",
        ))
        conn.commit()


def fetch_all_results():
    init_db()
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("""
            SELECT cv_filename, model_name, language, layout_type,
                   accuracy_pct, time_seconds, cost_usd, json_valid, schema_valid, error, run_at
            FROM llm_parsing_test_results
            ORDER BY run_at DESC, cv_filename, model_name
        """)
        return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Load CV tasks: parsed text + ground truth JSON for each CV in DB
# ---------------------------------------------------------------------------
def _get_cv_parsing_tasks():
    """Build list of { cv_filename, text, ground_truth_dict, metadata_row } for each CV that has both parsed .txt and json_parsed .json."""
    tasks = []
    for row in ocr.load_db_metadata():
        filename = row.get("filename", "")
        base, _ = os.path.splitext(filename)
        base = base.strip()
        txt_path = os.path.join(ocr.PARSING_TXT_DIR, f"{base}.txt")
        json_path = os.path.join(ocr.PARSING_DIR, f"{base}.json")
        if not (os.path.isfile(txt_path) and os.path.isfile(json_path)):
            continue
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()
            with open(json_path, "r", encoding="utf-8") as g:
                ground_truth = json.load(g)
        except Exception:
            continue
        if not (text and text.strip()):
            continue
        tasks.append({
            "cv_filename": filename,
            "text": text,
            "ground_truth": ground_truth,
            "metadata_row": row,
        })
    return tasks


# ---------------------------------------------------------------------------
# Run parsing for one CV (all models in parallel)
# ---------------------------------------------------------------------------
def _run_one_cv(
    cv_filename: str,
    text: str,
    ground_truth: dict,
    metadata_row: dict,
    replicate_key: str,
    results: list,
):
    """Run Gemini 4o nano, Claude, Gemini 2.5 Flash for one CV (parallel); append results to DB."""
    language = (metadata_row.get("language") or "") or ""
    layout_type = (metadata_row.get("layout_type") or "") or ""
    model_ids = list(PARSING_MODELS.keys())
    result_holder = [None] * len(model_ids)
    threads = []
    for i, model_id in enumerate(model_ids):
        t = Thread(
            target=_run_one_model,
            args=(model_id, text, None, replicate_key, result_holder, i),
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    for r in result_holder:
        if r is None:
            continue
        label = r.get("label", r.get("model_id", ""))
        accuracy_pct = None
        if ground_truth and r.get("parsed"):
            accuracy_pct = _accuracy_vs_ground_truth(r["parsed"], ground_truth)
        insert_result(
            cv_filename,
            label,
            language,
            layout_type,
            accuracy_pct,
            r.get("time_seconds", 0.0),
            r.get("cost_usd", 0.0),
            r.get("json_valid", False),
            r.get("schema_valid", False),
            r.get("error"),
        )
    results.append({"cv": cv_filename})


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Parallel LLM Parsing Test", layout="wide")
st.title("Parallel LLM Parsing Test: Gemini 4o nano, Claude, Gemini 2.5 Flash")
st.markdown(
    "Run **Gemini 4o nano**, **Claude 4.5 Haiku**, and **Gemini 2.5 Flash** on all CVs in the database. "
    "Input: parsed text from `ground_truth_database/parsed/`. Ground truth JSON from `ground_truth_database/json_parsed/`. "
    "Results are saved in **SQLite** (`ground_truth_database/llm_parsing_test_results.db`) and persist after you close the app. "
    "*(Gemini 4o nano uses Replicate's Gemini 2.5 Flash; 2.0 Flash is not available on Replicate.)*"
)

replicate_key = get_replicate_key()
if not replicate_key:
    st.error("Missing **REPLICATE_API_KEY** (or **REPLICATE_API_TOKEN**) in `.streamlit/secrets.toml` (or env).")
    st.stop()

cv_tasks = _get_cv_parsing_tasks()
if not cv_tasks:
    st.warning(
        "No CVs found with both `parsed/cvXXX.txt` and `json_parsed/cvXXX.json`. "
        "Add transcriptions and JSON ground truth to the ground truth database."
    )
else:
    run_all = st.button("Run LLM parsing on all CVs", type="primary", key="run_all_llm_parsing")
    if run_all and cv_tasks:
        progress = st.progress(0.0, text="Starting…")
        results_holder = []
        n = len(cv_tasks)
        for i, task in enumerate(cv_tasks):
            progress.progress((i + 1) / n, text=f"Running {task['cv_filename']} ({i + 1}/{n})…")
            _run_one_cv(
                task["cv_filename"],
                task["text"],
                task["ground_truth"],
                task["metadata_row"],
                replicate_key,
                results_holder,
            )
        progress.progress(1.0, text="Done.")
        st.success(f"Finished. Ran LLM parsing on {n} CVs (3 models each). Results saved to SQLite.")
        time.sleep(0.5)
        st.rerun()

# ---------- Results from SQLite ----------
st.divider()
st.subheader("Results")
with st.expander("Why do I see errors in the table?"):
    st.markdown("""
- **The read operation timed out** — The model took longer than the wait limit (now 120s). Slow CVs or Replicate load can cause this; re-run or try again later.
- **No valid JSON found in response** — The model returned text that wasn’t valid JSON (e.g. markdown or commentary). We try to extract JSON from code blocks or from the first `{` to the last `}`.
- **Additional properties are not allowed** — The model added extra keys (e.g. `additionalProperties`) to the JSON. We now strip unknown keys before validation, so this should appear less often.
""")
init_db()
rows = fetch_all_results()
if not rows:
    st.info(
        "No results yet. Click **Run LLM parsing on all CVs** to run Gemini 4o nano, Claude, and Gemini 2.5 Flash "
        "on every CV that has both parsed text and ground truth JSON."
    )
else:
    import pandas as pd

    df = pd.DataFrame(rows)
    ok = df[df["error"].isna()]

    st.markdown("### Summary: per-model averages")
    if not ok.empty:
        agg = ok.groupby("model_name").agg({
            "accuracy_pct": "mean",
            "time_seconds": "mean",
            "cost_usd": "sum",
            "json_valid": "mean",
            "schema_valid": "mean",
        }).round(2)
        agg.columns = ["Avg accuracy %", "Avg time (s)", "Total cost ($)", "JSON valid %", "Schema valid %"]
        st.dataframe(agg, use_container_width=True)

        # Verdict: who wins on accuracy, speed, cost
        model_names = ok["model_name"].unique().tolist()
        if len(model_names) >= 2:
            parts = []
            for m in model_names:
                sub = ok[ok["model_name"] == m]
                acc = sub["accuracy_pct"].mean()
                t = sub["time_seconds"].mean()
                c = sub["cost_usd"].sum()
                parts.append((m, acc, t, c))
            best_acc = max(parts, key=lambda x: (x[1] or 0))
            best_time = min(parts, key=lambda x: x[2])
            best_cost = min(parts, key=lambda x: x[3])
            st.markdown("**Verdict**")
            st.write(
                f"- **Best accuracy:** {best_acc[0]} (avg {best_acc[1]:.1f}%)  \n"
                f"- **Fastest:** {best_time[0]} (avg {best_time[2]:.2f}s)  \n"
                f"- **Cheapest (total):** {best_cost[0]} (${best_cost[3]:.4f})"
            )
    else:
        st.warning("No successful runs to compare; all rows had errors.")

    # Plots
    st.markdown("### Plots")
    try:
        import plotly.express as px
    except ImportError:
        st.caption("Install plotly for charts: `pip install plotly`")
    else:
        if not ok.empty:
            col1, col2 = st.columns(2)
            with col1:
                by_layout = ok.groupby(["layout_type", "model_name"]).agg({"accuracy_pct": "mean"}).reset_index()
                if not by_layout.empty:
                    fig = px.bar(
                        by_layout, x="layout_type", y="accuracy_pct", color="model_name", barmode="group",
                        title="Average accuracy % by layout type",
                        labels={"accuracy_pct": "Avg accuracy %", "layout_type": "Layout"},
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                by_lang = ok.groupby(["language", "model_name"]).agg({"time_seconds": "mean"}).reset_index()
                if not by_lang.empty:
                    fig2 = px.bar(
                        by_lang, x="language", y="time_seconds", color="model_name", barmode="group",
                        title="Average time (s) by language",
                        labels={"time_seconds": "Avg time (s)", "language": "Language"},
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            overall = ok.groupby("model_name").agg({"time_seconds": "mean", "accuracy_pct": "mean"}).reset_index()
            fig3 = px.bar(overall, x="model_name", y="time_seconds", title="Average time per run (seconds)", labels={"time_seconds": "Avg time (s)", "model_name": "Model"})
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### All runs")
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv_data = df.to_csv(index=False)
    st.download_button("Download results as CSV", csv_data, file_name="llm_parsing_test_results.csv", mime="text/csv", key="dl_csv")
