"""
Results summary: load OCR and LLM parsing test results from SQLite, show dashboard
metrics and charts, and recommend the best OCR and best LLM to pick.
"""
import os
import sqlite3
import sys
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

# Project root and ocr_common for DB paths
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_this_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)
import ocr_common as ocr

OCR_DB = os.path.join(ocr.DB_DIR, "ocr_test_results.db")
LLM_DB = os.path.join(ocr.DB_DIR, "llm_parsing_test_results.db")


def load_ocr_results():
    """Load OCR test results; init DB if missing."""
    if not os.path.isfile(OCR_DB):
        return pd.DataFrame()
    conn = sqlite3.connect(OCR_DB)
    df = pd.read_sql_query(
        """SELECT cv_filename, model_name, language, layout_type, extension, is_scanned,
                  cer_pct, wer_pct, layout_accuracy_pct, time_seconds, cost_usd, error, run_at
           FROM ocr_test_results
           ORDER BY run_at DESC""",
        conn,
    )
    conn.close()
    return df


def load_llm_results():
    """Load LLM parsing test results; init DB if missing."""
    if not os.path.isfile(LLM_DB):
        return pd.DataFrame()
    conn = sqlite3.connect(LLM_DB)
    df = pd.read_sql_query(
        """SELECT cv_filename, model_name, language, layout_type,
                  accuracy_pct, time_seconds, cost_usd, json_valid, schema_valid, error, run_at
           FROM llm_parsing_test_results
           ORDER BY run_at DESC""",
        conn,
    )
    conn.close()
    return df


def recommend_best_ocr(df: pd.DataFrame) -> Tuple[Optional[str], dict]:
    """Return (best_model_name, metrics_dict). Uses CER/WER/layout/speed/cost."""
    ok = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    if ok.empty:
        return None, {}
    by_model = (
        ok.groupby("model_name")
        .agg(
            {
                "cer_pct": "mean",
                "wer_pct": "mean",
                "layout_accuracy_pct": "mean",
                "time_seconds": "mean",
                "cost_usd": "sum",
            }
        )
        .reset_index()
    )
    # Best = lowest CER (primary), then lowest WER, then highest layout_accuracy
    best_idx = by_model.sort_values(
        by=["cer_pct", "wer_pct", "layout_accuracy_pct"],
        ascending=[True, True, False],
    ).index[0]
    best_row = by_model.loc[best_idx]
    metrics = by_model.set_index("model_name").to_dict("index")
    return str(best_row["model_name"]), metrics


def recommend_best_llm(df: pd.DataFrame) -> Tuple[Optional[str], dict]:
    """Return (best_model_name, metrics_dict). Uses accuracy first, then json_valid, schema_valid, cost."""
    ok = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")]
    if ok.empty:
        return None, {}
    by_model = (
        ok.groupby("model_name")
        .agg(
            {
                "accuracy_pct": "mean",
                "json_valid": "mean",
                "schema_valid": "mean",
                "time_seconds": "mean",
                "cost_usd": "sum",
            }
        )
        .reset_index()
    )
    # Best = highest accuracy, then highest json_valid, then schema_valid
    best_idx = by_model.sort_values(
        by=["accuracy_pct", "json_valid", "schema_valid"],
        ascending=[False, False, False],
    ).index[0]
    best_row = by_model.loc[best_idx]
    metrics = by_model.set_index("model_name").to_dict("index")
    return str(best_row["model_name"]), metrics


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Results Summary", layout="wide")
st.title("Results Summary: OCR & LLM Parsing")
st.markdown(
    "Data loaded from **OCR test** (`ground_truth_database/ocr_test_results.db`) and "
    "**LLM parsing test** (`ground_truth_database/llm_parsing_test_results.db`). "
    "Use the metrics and charts below to choose the best OCR and best LLM."
)

ocr_df = load_ocr_results()
llm_df = load_llm_results()

# ---------- Final recommendations (top) ----------
best_ocr, ocr_metrics = recommend_best_ocr(ocr_df) if not ocr_df.empty else (None, {})
best_llm, llm_metrics = recommend_best_llm(llm_df) if not llm_df.empty else (None, {})

rec1, rec2 = st.columns(2)
with rec1:
    if best_ocr:
        st.success(f"**Recommended OCR** — **{best_ocr}**")
        if ocr_metrics and best_ocr in ocr_metrics:
            m = ocr_metrics[best_ocr]
            st.caption(
                f"Avg CER: {m.get('cer_pct', 0):.2f}% · Avg WER: {m.get('wer_pct', 0):.2f}% · "
                f"Layout acc: {m.get('layout_accuracy_pct', 0):.1f}% · "
                f"Avg time: {m.get('time_seconds', 0):.2f}s · Total cost: ${m.get('cost_usd', 0):.4f}"
            )
    else:
        st.info("**Recommended OCR** — Run the *Parallel OCR Test* to get results.")
with rec2:
    if best_llm:
        st.success(f"**Recommended LLM (parsing)** — **{best_llm}**")
        if llm_metrics and best_llm in llm_metrics:
            m = llm_metrics[best_llm]
            st.caption(
                f"Avg accuracy: {m.get('accuracy_pct', 0):.1f}% · JSON valid: {m.get('json_valid', 0)*100:.0f}% · "
                f"Schema valid: {m.get('schema_valid', 0)*100:.0f}% · "
                f"Avg time: {m.get('time_seconds', 0):.2f}s · Total cost: ${m.get('cost_usd', 0):.4f}"
            )
    else:
        st.info("**Recommended LLM** — Run the *Parallel LLM Parsing Test* to get results.")

st.divider()

# ---------- OCR section ----------
st.header("OCR Test Results")
if ocr_df.empty:
    st.warning("No OCR results yet. Run **Parallel OCR Test** and then come back.")
else:
    ocr_ok = ocr_df[ocr_df["error"].isna() | (ocr_df["error"].astype(str).str.strip() == "")]
    n_ocr = len(ocr_ok)
    n_ocr_total = len(ocr_df)

    st.metric("Successful runs", f"{n_ocr} / {n_ocr_total}")

    if not ocr_ok.empty:
        # KPIs by model
        by_model = (
            ocr_ok.groupby("model_name")
            .agg(
                cer_pct=("cer_pct", "mean"),
                wer_pct=("wer_pct", "mean"),
                layout_accuracy_pct=("layout_accuracy_pct", "mean"),
                time_seconds=("time_seconds", "mean"),
                cost_usd=("cost_usd", "sum"),
                runs=("cv_filename", "count"),
            )
            .reset_index()
        )

        st.markdown("**Metrics by model**")
        summary = by_model[["model_name", "cer_pct", "wer_pct", "layout_accuracy_pct", "time_seconds", "cost_usd"]].copy()
        summary.columns = ["Model", "Avg CER %", "Avg WER %", "Layout acc %", "Avg time (s)", "Total cost (USD)"]
        summary["Avg CER %"] = summary["Avg CER %"].round(2)
        summary["Avg WER %"] = summary["Avg WER %"].round(2)
        summary["Layout acc %"] = summary["Layout acc %"].round(1)
        summary["Avg time (s)"] = summary["Avg time (s)"].round(2)
        summary["Total cost (USD)"] = summary["Total cost (USD)"].round(4)
        st.dataframe(summary, use_container_width=True, hide_index=True)

        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError:
            st.caption("Install plotly for charts: `pip install plotly`")
        else:
            st.subheader("OCR — Charts")
            col_a, col_b = st.columns(2)
            with col_a:
                by_layout = (
                    ocr_ok.groupby(["layout_type", "model_name"])
                    .agg({"cer_pct": "mean"})
                    .reset_index()
                )
                if not by_layout.empty:
                    fig = px.bar(
                        by_layout,
                        x="layout_type",
                        y="cer_pct",
                        color="model_name",
                        barmode="group",
                        title="Avg CER % by layout type (lower is better)",
                        labels={"cer_pct": "Avg CER %", "layout_type": "Layout"},
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with col_b:
                by_lang = (
                    ocr_ok.groupby(["language", "model_name"])
                    .agg({"cer_pct": "mean"})
                    .reset_index()
                )
                if not by_lang.empty:
                    fig2 = px.bar(
                        by_lang,
                        x="language",
                        y="cer_pct",
                        color="model_name",
                        barmode="group",
                        title="Avg CER % by language (lower is better)",
                        labels={"cer_pct": "Avg CER %", "language": "Language"},
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            col_c, col_d = st.columns(2)
            with col_c:
                overall_cer = ocr_ok.groupby("model_name")["cer_pct"].mean().reset_index()
                fig3 = px.bar(
                    overall_cer,
                    x="model_name",
                    y="cer_pct",
                    title="Average CER % by model (lower is better)",
                    labels={"cer_pct": "Avg CER %", "model_name": "Model"},
                )
                st.plotly_chart(fig3, use_container_width=True)
            with col_d:
                overall_layout = ocr_ok.groupby("model_name")["layout_accuracy_pct"].mean().reset_index()
                fig4 = px.bar(
                    overall_layout,
                    x="model_name",
                    y="layout_accuracy_pct",
                    title="Average layout accuracy % by model (higher is better)",
                    labels={"layout_accuracy_pct": "Layout accuracy %", "model_name": "Model"},
                )
                st.plotly_chart(fig4, use_container_width=True)

            col_e, col_f = st.columns(2)
            with col_e:
                overall_time = ocr_ok.groupby("model_name")["time_seconds"].mean().reset_index()
                fig5 = px.bar(
                    overall_time,
                    x="model_name",
                    y="time_seconds",
                    title="Average time per run (seconds)",
                    labels={"time_seconds": "Avg time (s)", "model_name": "Model"},
                )
                st.plotly_chart(fig5, use_container_width=True)
            with col_f:
                total_cost = ocr_ok.groupby("model_name")["cost_usd"].sum().reset_index()
                fig6 = px.bar(
                    total_cost,
                    x="model_name",
                    y="cost_usd",
                    title="Total cost per model (USD)",
                    labels={"cost_usd": "Cost (USD)", "model_name": "Model"},
                )
                st.plotly_chart(fig6, use_container_width=True)

    with st.expander("OCR raw data"):
        st.dataframe(ocr_df, use_container_width=True, hide_index=True)

st.divider()

# ---------- LLM section ----------
st.header("LLM Parsing Test Results")
if llm_df.empty:
    st.warning("No LLM parsing results yet. Run **Parallel LLM Parsing Test** and then come back.")
else:
    llm_ok = llm_df[llm_df["error"].isna() | (llm_df["error"].astype(str).str.strip() == "")]
    n_llm = len(llm_ok)
    n_llm_total = len(llm_df)

    st.metric("Successful runs", f"{n_llm} / {n_llm_total}")

    if not llm_ok.empty:
        by_model_llm = (
            llm_ok.groupby("model_name")
            .agg(
                accuracy_pct=("accuracy_pct", "mean"),
                json_valid=("json_valid", "mean"),
                schema_valid=("schema_valid", "mean"),
                time_seconds=("time_seconds", "mean"),
                cost_usd=("cost_usd", "sum"),
                runs=("cv_filename", "count"),
            )
            .reset_index()
        )

        st.markdown("**Metrics by model**")
        sum_llm = by_model_llm[["model_name", "accuracy_pct", "json_valid", "schema_valid", "time_seconds", "cost_usd"]].copy()
        sum_llm["json_valid"] = (sum_llm["json_valid"] * 100).round(0)
        sum_llm["schema_valid"] = (sum_llm["schema_valid"] * 100).round(0)
        sum_llm.columns = ["Model", "Avg accuracy %", "JSON valid %", "Schema valid %", "Avg time (s)", "Total cost (USD)"]
        sum_llm["Avg accuracy %"] = sum_llm["Avg accuracy %"].round(1)
        sum_llm["Avg time (s)"] = sum_llm["Avg time (s)"].round(2)
        sum_llm["Total cost (USD)"] = sum_llm["Total cost (USD)"].round(4)
        st.dataframe(sum_llm, use_container_width=True, hide_index=True)

        try:
            import plotly.express as px
        except ImportError:
            st.caption("Install plotly for charts: `pip install plotly`")
        else:
            st.subheader("LLM Parsing — Charts")
            col1, col2 = st.columns(2)
            with col1:
                acc_by_model = llm_ok.groupby("model_name")["accuracy_pct"].mean().reset_index()
                fig = px.bar(
                    acc_by_model,
                    x="model_name",
                    y="accuracy_pct",
                    title="Average parsing accuracy % by model (higher is better)",
                    labels={"accuracy_pct": "Avg accuracy %", "model_name": "Model"},
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                valid_rates = (
                    llm_ok.groupby("model_name")
                    .agg({"json_valid": "mean", "schema_valid": "mean"})
                    .reset_index()
                )
                import plotly.graph_objects as go
                fig_v = go.Figure(
                    data=[
                        go.Bar(name="JSON valid %", x=valid_rates["model_name"], y=valid_rates["json_valid"] * 100),
                        go.Bar(name="Schema valid %", x=valid_rates["model_name"], y=valid_rates["schema_valid"] * 100),
                    ]
                )
                fig_v.update_layout(
                    barmode="group",
                    title="JSON & schema validity rate by model",
                    xaxis_title="Model",
                    yaxis_title="%",
                )
                st.plotly_chart(fig_v, use_container_width=True)

            by_layout_llm = (
                llm_ok.groupby(["layout_type", "model_name"])
                .agg({"accuracy_pct": "mean"})
                .reset_index()
            )
            if not by_layout_llm.empty:
                fig_l = px.bar(
                    by_layout_llm,
                    x="layout_type",
                    y="accuracy_pct",
                    color="model_name",
                    barmode="group",
                    title="Avg accuracy % by layout type",
                    labels={"accuracy_pct": "Avg accuracy %", "layout_type": "Layout"},
                )
                st.plotly_chart(fig_l, use_container_width=True)

            col_t, col_c = st.columns(2)
            with col_t:
                time_llm = llm_ok.groupby("model_name")["time_seconds"].mean().reset_index()
                fig_t = px.bar(
                    time_llm,
                    x="model_name",
                    y="time_seconds",
                    title="Average time per run (seconds)",
                    labels={"time_seconds": "Avg time (s)", "model_name": "Model"},
                )
                st.plotly_chart(fig_t, use_container_width=True)
            with col_c:
                cost_llm = llm_ok.groupby("model_name")["cost_usd"].sum().reset_index()
                fig_c = px.bar(
                    cost_llm,
                    x="model_name",
                    y="cost_usd",
                    title="Total cost per model (USD)",
                    labels={"cost_usd": "Cost (USD)", "model_name": "Model"},
                )
                st.plotly_chart(fig_c, use_container_width=True)

    with st.expander("LLM parsing raw data"):
        st.dataframe(llm_df, use_container_width=True, hide_index=True)

st.divider()

# ---------- Final recommendation summary ----------
st.header("Final recommendation")
rec_col1, rec_col2 = st.columns(2)
with rec_col1:
    st.markdown("**Best OCR to pick**")
    if best_ocr:
        st.success(f"**{best_ocr}** — best trade-off on CER, WER, layout accuracy, speed and cost.")
    else:
        st.info("Run the Parallel OCR Test first.")
with rec_col2:
    st.markdown("**Best LLM to pick (for CV parsing)**")
    if best_llm:
        st.success(f"**{best_llm}** — best accuracy with valid JSON/schema and reasonable cost.")
    else:
        st.info("Run the Parallel LLM Parsing Test first.")
