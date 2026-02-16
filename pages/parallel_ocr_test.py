"""
Parallel OCR test: run Mistral OCR 3 and Replicate text-extract-ocr on all CVs in the database.
Results are stored in SQLite (database/ocr_test_results.db) and persist after closing the app.
Plots: by layout_type, by language, overall. Summary: which model is better.
"""
import os
import sqlite3
import time
from datetime import datetime
from threading import Thread

import streamlit as st

# Import shared OCR logic (project root)
import sys
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
import ocr_common as ocr

# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(ocr.DB_DIR, "ocr_test_results.db")
MODEL_MISTRAL = "Mistral OCR 3"
MODEL_REPLICATE = "Replicate text-extract-ocr"


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ocr_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cv_filename TEXT NOT NULL,
                model_name TEXT NOT NULL,
                language TEXT,
                layout_type TEXT,
                extension TEXT,
                is_scanned INTEGER,
                cer_pct REAL,
                wer_pct REAL,
                layout_accuracy_pct REAL,
                time_seconds REAL,
                cost_usd REAL,
                error TEXT,
                run_at TEXT
            )
        """)
        conn.commit()


def insert_result(cv_filename: str, model_name: str, language: str, layout_type: str, extension: str,
                  is_scanned: int, cer_pct: float, wer_pct: float, layout_accuracy_pct: float,
                  time_seconds: float, cost_usd: float, error: str):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO ocr_test_results
            (cv_filename, model_name, language, layout_type, extension, is_scanned,
             cer_pct, wer_pct, layout_accuracy_pct, time_seconds, cost_usd, error, run_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (cv_filename, model_name, language, layout_type, extension, is_scanned,
              cer_pct, wer_pct, layout_accuracy_pct, time_seconds, cost_usd, error,
              datetime.utcnow().isoformat() + "Z"))
        conn.commit()


def fetch_all_results():
    init_db()
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("""
            SELECT cv_filename, model_name, language, layout_type, extension, is_scanned,
                   cer_pct, wer_pct, layout_accuracy_pct, time_seconds, cost_usd, error, run_at
            FROM ocr_test_results
            ORDER BY run_at DESC, cv_filename, model_name
        """)
        return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Run OCR for one CV (both models in parallel)
# ---------------------------------------------------------------------------
def _run_one_cv(cv_filename: str, pdf_bytes: bytes, api_filename: str, ground_truth: str,
                metadata_row: dict, mistral_key: str, replicate_key: str, results: list):
    """Run Mistral and Replicate for one CV; append two result dicts to results (with metrics + metadata)."""
    language = metadata_row.get("language", "") or ""
    layout_type = metadata_row.get("layout_type", "") or ""
    extension = metadata_row.get("extension", "") or ""
    is_scanned = 1 if (metadata_row.get("is_scanned") in ("1", 1, True)) else 0

    out_mistral = [None]
    out_replicate = [None]

    def do_mistral():
        r = ocr.run_mistral_ocr(pdf_bytes, api_filename, mistral_key)
        out_mistral[0] = r

    def do_replicate():
        r = ocr.run_replicate_text_extract_ocr(pdf_bytes, replicate_key)
        out_replicate[0] = r

    t1 = Thread(target=do_mistral)
    t2 = Thread(target=do_replicate)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    for model_name, raw in [(MODEL_MISTRAL, out_mistral[0]), (MODEL_REPLICATE, out_replicate[0])]:
        if raw is None:
            continue
        text = (raw.get("text") or "").strip()
        err = raw.get("error")
        if err:
            insert_result(cv_filename, model_name, language, layout_type, extension, is_scanned,
                          0.0, 0.0, 0.0, raw.get("time_seconds", 0.0), raw.get("cost_usd", 0.0), err)
        else:
            m = ocr.word_metrics(ground_truth, text)
            insert_result(cv_filename, model_name, language, layout_type, extension, is_scanned,
                         m["cer_pct"], m["wer_pct"], m["layout_accuracy_pct"],
                         raw.get("time_seconds", 0.0), raw.get("cost_usd", 0.0), None)
    results.append({"cv": cv_filename})


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Parallel OCR Test", layout="wide")
st.title("Parallel OCR Test: Mistral vs Replicate text-extract-ocr")
st.markdown("Run **Mistral OCR 3** and **Replicate text-extract-ocr** on all CVs in the database. "
            "Results are saved in **SQLite** (`ground_truth_database/ocr_test_results.db`) and persist after you close the app.")

mistral_key = ocr.get_mistral_key()
replicate_key = ocr.get_replicate_key()
if not mistral_key:
    st.error("Missing **MISTRAL_API_KEY** in `.streamlit/secrets.toml` (or env).")
if not replicate_key:
    st.error("Missing **REPLICATE_API_KEY** or **REPLICATE_API_TOKEN** in `.streamlit/secrets.toml` (or env).")

if mistral_key and replicate_key:
    metadata = ocr.load_db_metadata()
    # Build list of CVs that have PDF + ground truth; track skipped for explanation
    cv_tasks = []
    skipped = []  # (filename, reason)
    for row in metadata:
        filename = row.get("filename", "")
        base, ext = os.path.splitext(filename)
        base = base.strip()
        txt_path = os.path.join(ocr.PARSING_TXT_DIR, f"{base}.txt")
        doc_path = os.path.join(ocr.CV_DIR, filename)
        if not (os.path.isfile(txt_path) and os.path.getsize(txt_path) > 0):
            skipped.append((filename, "no ground truth (missing or empty parsed/cvXXX.txt)"))
            continue
        if not os.path.isfile(doc_path):
            skipped.append((filename, "document missing in ground_truth_database/cv/"))
            continue
        pdf_bytes, api_filename, ground_truth = ocr.get_cv_pdf_and_ground_truth(row)
        if pdf_bytes is None or not ground_truth:
            skipped.append((filename, "document could not be read or converted to PDF"))
            continue
        cv_tasks.append({
            "metadata_row": row,
            "pdf_bytes": pdf_bytes,
            "api_filename": api_filename,
            "ground_truth": ground_truth,
            "cv_filename": filename,
        })

    if skipped:
        with st.expander(f"Why not all? ({len(skipped)} skipped)"):
            for fn, reason in skipped:
                st.markdown(f"- **{fn}**: {reason}")

    if not cv_tasks:
        st.warning("No CVs found with both a valid document (PDF/DOCX/PNG/JPG) and ground truth (e.g. `ground_truth_database/parsed/cv001.txt`). "
                   "Add files and transcriptions to the ground truth database.")

    run_all = st.button("Run OCR on all CVs", type="primary", key="run_all_ocr")
    if run_all and cv_tasks:
        progress = st.progress(0.0, text="Starting…")
        results_holder = []
        n = len(cv_tasks)
        for i, task in enumerate(cv_tasks):
            progress.progress((i + 1) / n, text=f"Running {task['cv_filename']} ({i + 1}/{n})…")
            _run_one_cv(
                task["cv_filename"],
                task["pdf_bytes"],
                task["api_filename"],
                task["ground_truth"],
                task["metadata_row"],
                mistral_key,
                replicate_key,
                results_holder,
            )
        progress.progress(1.0, text="Done.")
        st.success(f"Finished. Ran OCR on {n} CVs (2 models each). Results saved to SQLite.")
        time.sleep(0.5)
        st.rerun()

    # ---------- Results from SQLite ----------
    st.divider()
    st.subheader("Results")
    rows = fetch_all_results()
    if not rows:
        st.info("No results yet. Click **Run OCR on all CVs** to run Mistral OCR 3 and Replicate text-extract-ocr on every CV in the database.")
    else:
        import pandas as pd

        df = pd.DataFrame(rows)
        # Only consider rows without error for averages
        ok = df[df["error"].isna()]

        # Summary: per-model averages
        st.markdown("### Summary: who is better?")
        if not ok.empty:
            agg = ok.groupby("model_name").agg({
                "cer_pct": "mean",
                "wer_pct": "mean",
                "layout_accuracy_pct": "mean",
                "time_seconds": "mean",
                "cost_usd": "sum",
            }).round(2)
            agg.columns = ["Avg CER %", "Avg WER %", "Avg Layout %", "Avg time (s)", "Total cost ($)"]
            st.dataframe(agg, use_container_width=True)

            mistral_rows = ok[ok["model_name"] == MODEL_MISTRAL]
            rep_rows = ok[ok["model_name"] == MODEL_REPLICATE]
            if not mistral_rows.empty and not rep_rows.empty:
                m_cer = mistral_rows["cer_pct"].mean()
                r_cer = rep_rows["cer_pct"].mean()
                m_wer = mistral_rows["wer_pct"].mean()
                r_wer = rep_rows["wer_pct"].mean()
                m_layout = mistral_rows["layout_accuracy_pct"].mean()
                r_layout = rep_rows["layout_accuracy_pct"].mean()
                m_time = mistral_rows["time_seconds"].mean()
                r_time = rep_rows["time_seconds"].mean()
                m_cost = mistral_rows["cost_usd"].sum()
                r_cost = rep_rows["cost_usd"].sum()

                better_cer = MODEL_MISTRAL if m_cer < r_cer else MODEL_REPLICATE
                better_wer = MODEL_MISTRAL if m_wer < r_wer else MODEL_REPLICATE
                better_layout = MODEL_MISTRAL if m_layout > r_layout else MODEL_REPLICATE
                faster = MODEL_MISTRAL if m_time < r_time else MODEL_REPLICATE
                cheaper = MODEL_MISTRAL if m_cost < r_cost else MODEL_REPLICATE

                st.markdown("**Verdict**")
                st.write(
                    f"- **CER (lower is better):** {better_cer} (Mistral: {m_cer:.1f}%, Replicate: {r_cer:.1f}%)  \n"
                    f"- **WER (lower is better):** {better_wer} (Mistral: {m_wer:.1f}%, Replicate: {r_wer:.1f}%)  \n"
                    f"- **Layout accuracy (higher is better):** {better_layout} (Mistral: {m_layout:.1f}%, Replicate: {r_layout:.1f}%)  \n"
                    f"- **Faster:** {faster} (Mistral: {m_time:.2f}s avg, Replicate: {r_time:.2f}s avg)  \n"
                    f"- **Cheaper:** {cheaper} (Mistral: ${m_cost:.4f} total, Replicate: ${r_cost:.4f} total)"
                )
                wins_m = sum([m_cer < r_cer, m_wer < r_wer, m_layout > r_layout, m_time < r_time, m_cost < r_cost])
                wins_r = 5 - wins_m
                if wins_m > wins_r:
                    st.success(f"**Overall:** **{MODEL_MISTRAL}** wins on {wins_m}/5 criteria (accuracy, layout, speed, cost).")
                elif wins_r > wins_m:
                    st.success(f"**Overall:** **{MODEL_REPLICATE}** wins on {wins_r}/5 criteria.")
                else:
                    st.info("**Overall:** Tie on criteria count; choose by preference (e.g. accuracy vs cost).")
        else:
            st.warning("No successful runs to compare; all rows had errors.")

        # Plots
        st.markdown("### Plots")
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            st.caption("Install plotly for charts: `pip install plotly`")
        else:
            if not ok.empty:
                col1, col2 = st.columns(2)
                with col1:
                    # Avg CER by layout_type
                    by_layout = ok.groupby(["layout_type", "model_name"]).agg({"cer_pct": "mean"}).reset_index()
                    if not by_layout.empty:
                        fig_layout = px.bar(by_layout, x="layout_type", y="cer_pct", color="model_name", barmode="group",
                                            title="Average CER % by layout type", labels={"cer_pct": "Avg CER %", "layout_type": "Layout"})
                        st.plotly_chart(fig_layout, use_container_width=True)
                with col2:
                    # Avg time by language
                    by_lang = ok.groupby(["language", "model_name"]).agg({"time_seconds": "mean"}).reset_index()
                    if not by_lang.empty:
                        fig_lang = px.bar(by_lang, x="language", y="time_seconds", color="model_name", barmode="group",
                                          title="Average time (s) by language", labels={"time_seconds": "Avg time (s)", "language": "Language"})
                        st.plotly_chart(fig_lang, use_container_width=True)

                # Overall avg time
                overall = ok.groupby("model_name")["time_seconds"].mean().reset_index()
                fig_overall = px.bar(overall, x="model_name", y="time_seconds", title="Average time per run (seconds)", labels={"time_seconds": "Avg time (s)", "model_name": "Model"})
                st.plotly_chart(fig_overall, use_container_width=True)

        # Full table
        st.markdown("### All runs")
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv_data = df.to_csv(index=False)
        st.download_button("Download results as CSV", csv_data, file_name="ocr_test_results.csv", mime="text/csv", key="dl_csv")

init_db()
