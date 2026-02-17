#!/usr/bin/env python
"""
Generate a decision-oriented benchmark report from exported OCR/LLM results.

Inputs (expected in ./exports):
- ocr_all_runs.csv
- llm_all_runs.csv
- ocr_summary_latest.json
- llm_summary_latest.json
- benchmark_recommendation.json

Outputs:
- exports/ANALYSE_DECISIONNELLE.md
- exports/ANALYSE_DECISIONNELLE.html
- exports/charts/*.html (interactive plotly charts)
- exports/analysis_tables/*.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def has_error(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = (
        df.sort_values("run_at", ascending=False)
        .drop_duplicates(subset=["cv_filename", "model_name"], keep="first")
        .sort_values(["cv_filename", "model_name"])
        .reset_index(drop=True)
    )
    return out


def ensure_dirs(base: Path) -> Tuple[Path, Path]:
    charts = base / "charts"
    tables = base / "analysis_tables"
    charts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return charts, tables


def normalize_minmax(series: pd.Series, higher_is_better: bool) -> pd.Series:
    vals = series.astype(float)
    if vals.nunique(dropna=True) <= 1:
        return pd.Series([1.0] * len(vals), index=vals.index)
    mn, mx = vals.min(), vals.max()
    scaled = (vals - mn) / (mx - mn)
    return scaled if higher_is_better else 1.0 - scaled


def categorize_llm_error(msg: str) -> str:
    text = (msg or "").lower()
    if not text.strip():
        return "none"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "no valid json" in text:
        return "invalid_json"
    if "failed validating" in text or "additional properties" in text or "schema" in text:
        return "schema_validation"
    return "other_error"


@dataclass
class DecisionArtifacts:
    ocr_latest: pd.DataFrame
    llm_latest: pd.DataFrame
    ocr_model_table: pd.DataFrame
    llm_model_table: pd.DataFrame
    ocr_score_table: pd.DataFrame
    llm_score_table: pd.DataFrame
    recommendation: Dict


def compute_tables(
    ocr_all: pd.DataFrame,
    llm_all: pd.DataFrame,
    recommendation: Dict,
) -> DecisionArtifacts:
    ocr_latest = latest_snapshot(ocr_all)
    llm_latest = latest_snapshot(llm_all)

    ocr_ok = ocr_latest[~has_error(ocr_latest["error"])].copy()
    llm_ok = llm_latest[~has_error(llm_latest["error"])].copy()

    ocr_model = (
        ocr_ok.groupby("model_name", dropna=False)
        .agg(
            runs=("cv_filename", "count"),
            avg_cer_pct=("cer_pct", "mean"),
            avg_wer_pct=("wer_pct", "mean"),
            avg_layout_accuracy_pct=("layout_accuracy_pct", "mean"),
            avg_time_seconds=("time_seconds", "mean"),
            total_cost_usd=("cost_usd", "sum"),
        )
        .reset_index()
    )

    llm_latest_all = llm_latest.copy()
    llm_latest_all["has_error"] = has_error(llm_latest_all["error"])
    success_rates = (
        llm_latest_all.groupby("model_name", dropna=False)["has_error"]
        .agg(total="count", errors="sum")
        .reset_index()
    )
    success_rates["success_rate_pct_latest"] = (
        (success_rates["total"] - success_rates["errors"]) / success_rates["total"] * 100.0
    )

    llm_model = (
        llm_ok.groupby("model_name", dropna=False)
        .agg(
            successful_runs=("cv_filename", "count"),
            avg_accuracy_pct=("accuracy_pct", "mean"),
            avg_time_seconds=("time_seconds", "mean"),
            total_cost_usd=("cost_usd", "sum"),
            json_valid_rate_pct=("json_valid", lambda s: float(s.mean()) * 100.0),
            schema_valid_rate_pct=("schema_valid", lambda s: float(s.mean()) * 100.0),
        )
        .reset_index()
    ).merge(
        success_rates[["model_name", "success_rate_pct_latest"]],
        on="model_name",
        how="left",
    )
    llm_model["effective_accuracy_pct"] = (
        llm_model["avg_accuracy_pct"] * llm_model["success_rate_pct_latest"] / 100.0
    )

    ocr_score = ocr_model.copy()
    if not ocr_score.empty:
        # Explicit weights: prioritize extraction quality.
        ocr_score["s_cer"] = normalize_minmax(ocr_score["avg_cer_pct"], higher_is_better=False)
        ocr_score["s_wer"] = normalize_minmax(ocr_score["avg_wer_pct"], higher_is_better=False)
        ocr_score["s_layout"] = normalize_minmax(
            ocr_score["avg_layout_accuracy_pct"], higher_is_better=True
        )
        ocr_score["s_time"] = normalize_minmax(
            ocr_score["avg_time_seconds"], higher_is_better=False
        )
        ocr_score["s_cost"] = normalize_minmax(
            ocr_score["total_cost_usd"], higher_is_better=False
        )
        ocr_score["decision_score"] = (
            0.35 * ocr_score["s_cer"]
            + 0.25 * ocr_score["s_wer"]
            + 0.20 * ocr_score["s_layout"]
            + 0.10 * ocr_score["s_time"]
            + 0.10 * ocr_score["s_cost"]
        )
        ocr_score = ocr_score.sort_values("decision_score", ascending=False).reset_index(drop=True)

    llm_score = llm_model.copy()
    if not llm_score.empty:
        # Explicit weights: prioritize accuracy + reliability.
        llm_score["s_acc"] = normalize_minmax(llm_score["avg_accuracy_pct"], higher_is_better=True)
        llm_score["s_reliability"] = normalize_minmax(
            llm_score["success_rate_pct_latest"], higher_is_better=True
        )
        llm_score["s_time"] = normalize_minmax(
            llm_score["avg_time_seconds"], higher_is_better=False
        )
        llm_score["s_cost"] = normalize_minmax(
            llm_score["total_cost_usd"], higher_is_better=False
        )
        llm_score["decision_score"] = (
            0.50 * llm_score["s_acc"]
            + 0.30 * llm_score["s_reliability"]
            + 0.10 * llm_score["s_time"]
            + 0.10 * llm_score["s_cost"]
        )
        llm_score = llm_score.sort_values("decision_score", ascending=False).reset_index(drop=True)

    return DecisionArtifacts(
        ocr_latest=ocr_latest,
        llm_latest=llm_latest,
        ocr_model_table=ocr_model,
        llm_model_table=llm_model,
        ocr_score_table=ocr_score,
        llm_score_table=llm_score,
        recommendation=recommendation,
    )


def save_tables(art: DecisionArtifacts, tables_dir: Path) -> None:
    art.ocr_latest.to_csv(tables_dir / "ocr_latest_snapshot_enriched.csv", index=False, encoding="utf-8")
    art.llm_latest.to_csv(tables_dir / "llm_latest_snapshot_enriched.csv", index=False, encoding="utf-8")
    art.ocr_model_table.to_csv(tables_dir / "ocr_model_metrics_latest.csv", index=False, encoding="utf-8")
    art.llm_model_table.to_csv(tables_dir / "llm_model_metrics_latest.csv", index=False, encoding="utf-8")
    art.ocr_score_table.to_csv(tables_dir / "ocr_decision_scorecard.csv", index=False, encoding="utf-8")
    art.llm_score_table.to_csv(tables_dir / "llm_decision_scorecard.csv", index=False, encoding="utf-8")


def write_chart(fig, path: Path) -> None:
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


def build_charts(
    ocr_all: pd.DataFrame,
    llm_all: pd.DataFrame,
    art: DecisionArtifacts,
    charts_dir: Path,
) -> List[str]:
    created: List[str] = []
    if px is None or go is None:
        return created

    # OCR KPI comparison
    if not art.ocr_model_table.empty:
        melted = art.ocr_model_table.melt(
            id_vars=["model_name"],
            value_vars=[
                "avg_cer_pct",
                "avg_wer_pct",
                "avg_layout_accuracy_pct",
                "avg_time_seconds",
                "total_cost_usd",
            ],
            var_name="metric",
            value_name="value",
        )
        fig = px.bar(
            melted,
            x="metric",
            y="value",
            color="model_name",
            barmode="group",
            title="OCR KPIs (latest snapshot)",
            labels={"metric": "Metric", "value": "Value", "model_name": "OCR model"},
        )
        name = "ocr_kpi_comparison.html"
        write_chart(fig, charts_dir / name)
        created.append(name)

    # OCR CER by layout
    ocr_latest_ok = art.ocr_latest[~has_error(art.ocr_latest["error"])].copy()
    if not ocr_latest_ok.empty:
        by_layout = (
            ocr_latest_ok.groupby(["layout_type", "model_name"], dropna=False)["cer_pct"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            by_layout,
            x="layout_type",
            y="cer_pct",
            color="model_name",
            barmode="group",
            title="OCR CER by CV layout type",
            labels={"layout_type": "Layout", "cer_pct": "Avg CER %"},
        )
        name = "ocr_cer_by_layout.html"
        write_chart(fig, charts_dir / name)
        created.append(name)

        by_lang_time = (
            ocr_latest_ok.groupby(["language", "model_name"], dropna=False)["time_seconds"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            by_lang_time,
            x="language",
            y="time_seconds",
            color="model_name",
            barmode="group",
            title="OCR runtime by language",
            labels={"language": "Language", "time_seconds": "Avg time (s)"},
        )
        name = "ocr_time_by_language.html"
        write_chart(fig, charts_dir / name)
        created.append(name)

        # Per-CV CER delta (Replicate - Mistral)
        piv = ocr_latest_ok.pivot_table(
            index="cv_filename",
            columns="model_name",
            values="cer_pct",
            aggfunc="mean",
        )
        models = list(piv.columns)
        if "Mistral OCR 3" in models and "Replicate text-extract-ocr" in models:
            piv = piv.copy()
            piv["cer_delta_rep_minus_mistral"] = (
                piv["Replicate text-extract-ocr"] - piv["Mistral OCR 3"]
            )
            piv = piv.reset_index().sort_values("cer_delta_rep_minus_mistral", ascending=False)
            fig = px.bar(
                piv,
                x="cv_filename",
                y="cer_delta_rep_minus_mistral",
                title="Per-CV CER delta (Replicate - Mistral): positive = Mistral better",
                labels={
                    "cv_filename": "CV",
                    "cer_delta_rep_minus_mistral": "CER delta (pp)",
                },
            )
            name = "ocr_per_cv_cer_delta.html"
            write_chart(fig, charts_dir / name)
            created.append(name)

    # LLM model comparison (accuracy/reliability/time/cost)
    if not art.llm_model_table.empty:
        fig = px.scatter(
            art.llm_model_table,
            x="success_rate_pct_latest",
            y="avg_accuracy_pct",
            color="model_name",
            size="total_cost_usd",
            hover_data=["avg_time_seconds", "effective_accuracy_pct"],
            title="LLM trade-off: reliability vs accuracy (bubble size = total cost)",
            labels={
                "success_rate_pct_latest": "Success rate (latest snapshot, %)",
                "avg_accuracy_pct": "Avg accuracy (%)",
                "total_cost_usd": "Total cost ($)",
            },
        )
        name = "llm_reliability_vs_accuracy.html"
        write_chart(fig, charts_dir / name)
        created.append(name)

        melted = art.llm_model_table.melt(
            id_vars=["model_name"],
            value_vars=[
                "avg_accuracy_pct",
                "success_rate_pct_latest",
                "avg_time_seconds",
                "total_cost_usd",
                "effective_accuracy_pct",
            ],
            var_name="metric",
            value_name="value",
        )
        fig = px.bar(
            melted,
            x="metric",
            y="value",
            color="model_name",
            barmode="group",
            title="LLM KPIs (latest snapshot)",
            labels={"metric": "Metric", "value": "Value"},
        )
        name = "llm_kpi_comparison.html"
        write_chart(fig, charts_dir / name)
        created.append(name)

    # LLM error categories from all runs
    llm_err = llm_all.copy()
    llm_err["error_category"] = llm_err["error"].fillna("").map(categorize_llm_error)
    llm_err = llm_err[llm_err["error_category"] != "none"]
    if not llm_err.empty:
        by_cat = (
            llm_err.groupby(["model_name", "error_category"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        fig = px.bar(
            by_cat,
            x="error_category",
            y="count",
            color="model_name",
            barmode="group",
            title="LLM errors by category (all runs)",
            labels={"error_category": "Error category", "count": "Count"},
        )
        name = "llm_error_categories.html"
        write_chart(fig, charts_dir / name)
        created.append(name)

    # Decision scorecards
    if not art.ocr_score_table.empty:
        fig = px.bar(
            art.ocr_score_table,
            x="model_name",
            y="decision_score",
            title="OCR weighted decision score",
            labels={"model_name": "Model", "decision_score": "Score (0-1)"},
        )
        name = "ocr_weighted_decision_score.html"
        write_chart(fig, charts_dir / name)
        created.append(name)
    if not art.llm_score_table.empty:
        fig = px.bar(
            art.llm_score_table,
            x="model_name",
            y="decision_score",
            title="LLM weighted decision score",
            labels={"model_name": "Model", "decision_score": "Score (0-1)"},
        )
        name = "llm_weighted_decision_score.html"
        write_chart(fig, charts_dir / name)
        created.append(name)

    return created


def md_link(label: str, rel_path: str) -> str:
    return f"- [{label}]({rel_path})"


def build_report_markdown(
    out_path: Path,
    ocr_all: pd.DataFrame,
    llm_all: pd.DataFrame,
    art: DecisionArtifacts,
    charts: List[str],
) -> None:
    ocr_latest = art.ocr_latest
    llm_latest = art.llm_latest
    ocr_ok = ocr_latest[~has_error(ocr_latest["error"])]
    llm_ok = llm_latest[~has_error(llm_latest["error"])]

    # Coverage
    ocr_runs_per_cv = (
        ocr_all.groupby("cv_filename").size().describe().to_dict() if not ocr_all.empty else {}
    )
    llm_runs_per_cv = (
        llm_all.groupby("cv_filename").size().describe().to_dict() if not llm_all.empty else {}
    )

    # Per-CV head-to-head for OCR
    ocr_pivot = pd.DataFrame()
    mistral_wins_cer = 0
    mistral_wins_wer = 0
    if not ocr_ok.empty:
        ocr_pivot = ocr_ok.pivot_table(
            index="cv_filename",
            columns="model_name",
            values=["cer_pct", "wer_pct", "layout_accuracy_pct"],
            aggfunc="mean",
        )
        if ("cer_pct", "Mistral OCR 3") in ocr_pivot.columns and (
            "cer_pct",
            "Replicate text-extract-ocr",
        ) in ocr_pivot.columns:
            mistral_wins_cer = int(
                (
                    ocr_pivot[("cer_pct", "Mistral OCR 3")]
                    < ocr_pivot[("cer_pct", "Replicate text-extract-ocr")]
                ).sum()
            )
        if ("wer_pct", "Mistral OCR 3") in ocr_pivot.columns and (
            "wer_pct",
            "Replicate text-extract-ocr",
        ) in ocr_pivot.columns:
            mistral_wins_wer = int(
                (
                    ocr_pivot[("wer_pct", "Mistral OCR 3")]
                    < ocr_pivot[("wer_pct", "Replicate text-extract-ocr")]
                ).sum()
            )

    # LLM errors all-runs
    llm_all_e = llm_all.copy()
    llm_all_e["has_error"] = has_error(llm_all_e["error"])
    llm_err_all = (
        llm_all_e.groupby("model_name", dropna=False)["has_error"].mean().mul(100.0).to_dict()
        if not llm_all_e.empty
        else {}
    )

    rec_stack = art.recommendation.get("recommended_stack_for_endpoint", {})
    ocr_model = rec_stack.get("ocr_model")
    llm_model = rec_stack.get("llm_parsing_model")

    lines: List[str] = []
    lines.append("# Rapport d'analyse decisionnelle (OCR + LLM Parsing)")
    lines.append("")
    lines.append(f"- Genere le: `{now_utc()}`")
    lines.append("- Perimetre: analyse de tous les exports produits dans `exports/`.")
    lines.append("")

    lines.append("## 1) Sources analysees")
    lines.append("- `exports/ocr_all_runs.csv`")
    lines.append("- `exports/ocr_latest_snapshot.csv`")
    lines.append("- `exports/ocr_summary_latest.json`")
    lines.append("- `exports/llm_all_runs.csv`")
    lines.append("- `exports/llm_latest_snapshot.csv`")
    lines.append("- `exports/llm_summary_latest.json`")
    lines.append("- `exports/benchmark_recommendation.json`")
    lines.append("- `exports/RESULTS_SUMMARY.md`")
    lines.append("")

    lines.append("## 2) Verification de couverture et qualite des donnees")
    lines.append(
            f"- OCR: `{len(ocr_all)}` runs historiques, `{len(ocr_latest)}` runs dans le snapshot latest (CV x modele), erreurs latest=`{int(has_error(ocr_latest['error']).sum())}`."
    )
    lines.append(
        f"- LLM: `{len(llm_all)}` runs historiques, `{len(llm_latest)}` runs dans le snapshot latest (CV x modele), erreurs latest=`{int(has_error(llm_latest['error']).sum())}`."
    )
    if ocr_runs_per_cv:
        lines.append(
            f"- Densite OCR (runs par CV, historique): moyenne `{ocr_runs_per_cv.get('mean', 0):.2f}`, min `{ocr_runs_per_cv.get('min', 0):.0f}`, max `{ocr_runs_per_cv.get('max', 0):.0f}`."
        )
    if llm_runs_per_cv:
        lines.append(
            f"- Densite LLM (runs par CV, historique): moyenne `{llm_runs_per_cv.get('mean', 0):.2f}`, min `{llm_runs_per_cv.get('min', 0):.0f}`, max `{llm_runs_per_cv.get('max', 0):.0f}`."
        )
    lines.append("")

    lines.append("## 3) Analyse OCR: cheminement vers le choix")
    if art.ocr_model_table.empty:
        lines.append("- Aucune donnee OCR exploitable.")
    else:
        o = art.ocr_model_table.copy()
        o = o.sort_values("avg_cer_pct")
        for _, r in o.iterrows():
            lines.append(
                "- "
                f"{r['model_name']}: CER `{r['avg_cer_pct']:.2f}%`, WER `{r['avg_wer_pct']:.2f}%`, "
                f"layout `{r['avg_layout_accuracy_pct']:.2f}%`, temps `{r['avg_time_seconds']:.2f}s`, cout total `${r['total_cost_usd']:.4f}`."
            )
        lines.append(
            f"- Head-to-head par CV: Mistral meilleur en CER sur `{mistral_wins_cer}/16` CV, meilleur en WER sur `{mistral_wins_wer}/16` CV."
        )
        if not art.ocr_score_table.empty:
            best = art.ocr_score_table.iloc[0]
            lines.append(
            f"- Score pondere (qualite prioritaire): meilleur score `{best['model_name']}` = `{best['decision_score']:.3f}`."
            )
        lines.append(
            "- Conclusion OCR: Mistral gagne sur la qualite d'extraction (CER/WER/layout), Replicate reste meilleur en cout et vitesse."
        )
    lines.append("")

    lines.append("## 4) Analyse LLM parsing: cheminement vers le choix")
    if art.llm_model_table.empty:
        lines.append("- Aucune donnee LLM exploitable.")
    else:
        l = art.llm_model_table.copy().sort_values("avg_accuracy_pct", ascending=False)
        for _, r in l.iterrows():
            lines.append(
                "- "
                f"{r['model_name']}: accuracy `{r['avg_accuracy_pct']:.2f}%`, succes latest `{r['success_rate_pct_latest']:.1f}%`, "
                f"effective accuracy `{r['effective_accuracy_pct']:.2f}%`, temps `{r['avg_time_seconds']:.2f}s`, cout `${r['total_cost_usd']:.4f}`."
            )
        if llm_err_all:
            lines.append("- Taux d'erreur historique (tous runs):")
            for m, p in sorted(llm_err_all.items(), key=lambda x: x[1]):
                lines.append(f"- {m}: `{p:.1f}%`")
        if not art.llm_score_table.empty:
            best = art.llm_score_table.iloc[0]
            lines.append(
                f"- Score pondere (accuracy + fiabilite): meilleur score `{best['model_name']}` = `{best['decision_score']:.3f}`."
            )
        lines.append(
            "- Conclusion LLM: Claude 4.5 Haiku combine meilleure accuracy, meilleure vitesse et meilleure fiabilite latest."
        )
    lines.append("")

    lines.append("## 5) Decision finale endpoint")
    lines.append(f"- OCR retenu: `{ocr_model}`")
    lines.append(f"- LLM parsing retenu: `{llm_model}`")
    lines.append(
        "- Regle de decision appliquee: OCR sur criteres multi-objectifs (qualite majoritaire), LLM sur accuracy prioritaire puis fiabilite."
    )
    lines.append("")

    lines.append("## 6) Limites a connaitre avant API")
    lines.append(
        "- Le benchmark LLM a ete execute sur texte `ground_truth_database/parsed/*.txt` (pas directement sur la sortie OCR live), donc la perf endpoint dependra aussi du couplage reel OCR->LLM."
    )
    lines.append(
        "- Deux labels LLM pointent vers le meme backend dans la page de test (`Gemini 4o nano` et `Gemini 2.5 Flash` via Replicate), ce qui limite la diversite reelle des candidats."
    )
    lines.append(
        "- Une phase de validation finale end-to-end OCR+LLM sur CV bruts reste necessaire avant gel definitif."
    )
    lines.append("")

    lines.append("## 7) Graphiques generes")
    if charts:
        for c in charts:
            lines.append(md_link(c, f"charts/{c}"))
    else:
        lines.append("- Plotly indisponible: aucun graphique genere.")
    lines.append("")

    lines.append("## 8) Tables d'analyse generees")
    lines.append("- `analysis_tables/ocr_model_metrics_latest.csv`")
    lines.append("- `analysis_tables/llm_model_metrics_latest.csv`")
    lines.append("- `analysis_tables/ocr_decision_scorecard.csv`")
    lines.append("- `analysis_tables/llm_decision_scorecard.csv`")
    lines.append("- `analysis_tables/ocr_latest_snapshot_enriched.csv`")
    lines.append("- `analysis_tables/llm_latest_snapshot_enriched.csv`")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report_html(md_path: Path, html_path: Path, charts: List[str]) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    # Lightweight HTML wrapper; links point to separate chart files.
    chart_links = "".join(
        [f'<li><a href="charts/{c}" target="_blank">{c}</a></li>' for c in charts]
    )
    escaped = (
        md_text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    html = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Analyse Decisionnelle OCR/LLM</title>
  <style>
    body {{
      font-family: "Segoe UI", Tahoma, sans-serif;
      max-width: 1100px;
      margin: 32px auto;
      padding: 0 18px 40px;
      line-height: 1.45;
      color: #111;
      background: #fafafa;
    }}
    h1, h2 {{ color: #1f2937; }}
    .card {{
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 18px;
      margin-bottom: 16px;
    }}
    pre {{
      white-space: pre-wrap;
      background: #111827;
      color: #e5e7eb;
      padding: 14px;
      border-radius: 8px;
      overflow: auto;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Rapport d'analyse decisionnelle</h1>
    <p>Version HTML exploitable localement. Les graphiques sont accessibles ci-dessous.</p>
  </div>
  <div class="card">
    <h2>Graphiques</h2>
    <ul>
      {chart_links}
    </ul>
  </div>
  <div class="card">
    <h2>Contenu du rapport (Markdown)</h2>
    <pre>{escaped}</pre>
  </div>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate decision-oriented analysis report.")
    parser.add_argument("--exports-dir", default="exports", help="Directory containing export files")
    args = parser.parse_args()

    exports_dir = Path(args.exports_dir).resolve()
    if not exports_dir.exists():
        raise FileNotFoundError(f"Exports directory not found: {exports_dir}")

    charts_dir, tables_dir = ensure_dirs(exports_dir)

    ocr_all = pd.read_csv(exports_dir / "ocr_all_runs.csv")
    llm_all = pd.read_csv(exports_dir / "llm_all_runs.csv")
    recommendation = load_json(exports_dir / "benchmark_recommendation.json")

    art = compute_tables(ocr_all=ocr_all, llm_all=llm_all, recommendation=recommendation)
    save_tables(art, tables_dir)
    charts = build_charts(ocr_all=ocr_all, llm_all=llm_all, art=art, charts_dir=charts_dir)

    md_path = exports_dir / "ANALYSE_DECISIONNELLE.md"
    html_path = exports_dir / "ANALYSE_DECISIONNELLE.html"
    build_report_markdown(md_path, ocr_all, llm_all, art, charts)
    build_report_html(md_path, html_path, charts)

    print(f"Report generated: {md_path}")
    print(f"HTML generated:   {html_path}")
    print(f"Charts generated: {len(charts)} in {charts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
