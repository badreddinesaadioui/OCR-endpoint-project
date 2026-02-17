#!/usr/bin/env python
"""
Export OCR and LLM benchmark results from SQLite to local files.

Outputs in ./exports by default:
- ocr_all_runs.csv
- ocr_latest_snapshot.csv
- ocr_summary_latest.json
- llm_all_runs.csv
- llm_latest_snapshot.csv
- llm_summary_latest.json
- benchmark_recommendation.json
- RESULTS_SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def no_error_mask(df: pd.DataFrame) -> pd.Series:
    if "error" not in df.columns:
        return pd.Series([True] * len(df), index=df.index)
    text = df["error"].fillna("").astype(str).str.strip()
    return text.eq("")


def load_table(db_path: Path, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    snapshot = (
        df.sort_values("run_at", ascending=False)
        .drop_duplicates(subset=["cv_filename", "model_name"], keep="first")
        .sort_values(["cv_filename", "model_name", "run_at"])
        .reset_index(drop=True)
    )
    return snapshot


def _round_records(records: List[dict], digits: int = 6) -> List[dict]:
    out: List[dict] = []
    for row in records:
        clean = {}
        for k, v in row.items():
            if isinstance(v, float):
                clean[k] = round(v, digits)
            else:
                clean[k] = v
        out.append(clean)
    return out


def summarize_ocr(df_latest: pd.DataFrame, df_all: pd.DataFrame) -> Dict:
    ok = df_latest[no_error_mask(df_latest)].copy()
    summary: Dict = {
        "generated_at": utc_now_iso(),
        "source_table": "ocr_test_results",
        "all_rows_count": int(len(df_all)),
        "latest_snapshot_rows_count": int(len(df_latest)),
        "latest_snapshot_success_rows_count": int(len(ok)),
        "latest_snapshot_error_rows_count": int(len(df_latest) - len(ok)),
        "models": [],
        "verdict": {},
    }

    if ok.empty:
        return summary

    agg = (
        ok.groupby("model_name", dropna=False)
        .agg(
            successful_runs=("cv_filename", "count"),
            avg_cer_pct=("cer_pct", "mean"),
            avg_wer_pct=("wer_pct", "mean"),
            avg_layout_accuracy_pct=("layout_accuracy_pct", "mean"),
            avg_time_seconds=("time_seconds", "mean"),
            total_cost_usd=("cost_usd", "sum"),
        )
        .reset_index()
    )
    summary["models"] = _round_records(agg.to_dict(orient="records"), digits=6)

    by_model = {row["model_name"]: row for row in summary["models"]}
    model_names = list(by_model.keys())
    if len(model_names) < 2:
        return summary

    # Same criteria style as Streamlit page:
    # - CER lower is better
    # - WER lower is better
    # - Layout accuracy higher is better
    # - Time lower is better
    # - Cost lower is better
    criteria_defs = [
        ("cer", "avg_cer_pct", "min"),
        ("wer", "avg_wer_pct", "min"),
        ("layout", "avg_layout_accuracy_pct", "max"),
        ("time", "avg_time_seconds", "min"),
        ("cost", "total_cost_usd", "min"),
    ]
    winners: Dict[str, Optional[str]] = {}
    wins_count: Dict[str, int] = {name: 0 for name in model_names}
    for label, field, mode in criteria_defs:
        values = {name: by_model[name].get(field) for name in model_names}
        values = {k: v for k, v in values.items() if v is not None}
        if not values:
            winners[label] = None
            continue
        target = min(values.values()) if mode == "min" else max(values.values())
        tied = [name for name, val in values.items() if val == target]
        if len(tied) == 1:
            winners[label] = tied[0]
            wins_count[tied[0]] += 1
        else:
            winners[label] = None

    best_wins = max(wins_count.values()) if wins_count else 0
    tied_global = [name for name, cnt in wins_count.items() if cnt == best_wins]
    overall_winner = tied_global[0] if len(tied_global) == 1 else None

    summary["verdict"] = {
        "criteria_winner": winners,
        "wins_count": wins_count,
        "overall_winner": overall_winner,
    }
    return summary


def summarize_llm(df_latest: pd.DataFrame, df_all: pd.DataFrame) -> Dict:
    ok = df_latest[no_error_mask(df_latest)].copy()
    errors = df_latest[~no_error_mask(df_latest)].copy()
    summary: Dict = {
        "generated_at": utc_now_iso(),
        "source_table": "llm_parsing_test_results",
        "all_rows_count": int(len(df_all)),
        "latest_snapshot_rows_count": int(len(df_latest)),
        "latest_snapshot_success_rows_count": int(len(ok)),
        "latest_snapshot_error_rows_count": int(len(errors)),
        "models": [],
        "errors_by_model": {},
        "verdict": {},
    }

    if not errors.empty:
        err_counts = errors.groupby("model_name").size().to_dict()
        summary["errors_by_model"] = {k: int(v) for k, v in err_counts.items()}

    if ok.empty:
        return summary

    agg = (
        ok.groupby("model_name", dropna=False)
        .agg(
            successful_runs=("cv_filename", "count"),
            avg_accuracy_pct=("accuracy_pct", "mean"),
            avg_time_seconds=("time_seconds", "mean"),
            total_cost_usd=("cost_usd", "sum"),
            json_valid_rate=("json_valid", "mean"),
            schema_valid_rate=("schema_valid", "mean"),
        )
        .reset_index()
    )
    if not agg.empty:
        agg["json_valid_rate"] = agg["json_valid_rate"] * 100.0
        agg["schema_valid_rate"] = agg["schema_valid_rate"] * 100.0
    summary["models"] = _round_records(agg.to_dict(orient="records"), digits=6)

    by_model = {row["model_name"]: row for row in summary["models"]}
    model_names = list(by_model.keys())
    if not model_names:
        return summary

    best_acc = max(
        model_names,
        key=lambda m: (by_model[m].get("avg_accuracy_pct") or float("-inf")),
    )
    fastest = min(
        model_names,
        key=lambda m: (by_model[m].get("avg_time_seconds") or float("inf")),
    )
    cheapest = min(
        model_names,
        key=lambda m: (by_model[m].get("total_cost_usd") or float("inf")),
    )

    summary["verdict"] = {
        "best_accuracy_model": best_acc,
        "fastest_model": fastest,
        "cheapest_model": cheapest,
    }
    return summary


def build_recommendation(ocr_summary: Dict, llm_summary: Dict) -> Dict:
    ocr_winner = ocr_summary.get("verdict", {}).get("overall_winner")
    if not ocr_winner:
        # fallback on minimum CER
        models = ocr_summary.get("models", [])
        if models:
            ocr_winner = min(models, key=lambda r: r.get("avg_cer_pct", float("inf"))).get("model_name")

    llm_winner = llm_summary.get("verdict", {}).get("best_accuracy_model")
    if not llm_winner:
        models = llm_summary.get("models", [])
        if models:
            llm_winner = max(models, key=lambda r: r.get("avg_accuracy_pct", float("-inf"))).get("model_name")

    return {
        "generated_at": utc_now_iso(),
        "recommended_stack_for_endpoint": {
            "ocr_model": ocr_winner,
            "llm_parsing_model": llm_winner,
            "selection_rule": "OCR by multi-criteria verdict (fallback CER), LLM by highest average accuracy.",
        },
        "ocr_verdict": ocr_summary.get("verdict", {}),
        "llm_verdict": llm_summary.get("verdict", {}),
    }


def write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_report(path: Path, ocr: Dict, llm: Dict, rec: Dict) -> None:
    lines: List[str] = []
    lines.append("# Benchmark Results Export")
    lines.append("")
    lines.append(f"- Generated at: `{utc_now_iso()}`")
    lines.append("")

    lines.append("## OCR")
    lines.append(f"- All rows: `{ocr.get('all_rows_count', 0)}`")
    lines.append(f"- Latest snapshot rows: `{ocr.get('latest_snapshot_rows_count', 0)}`")
    lines.append(f"- Successful rows: `{ocr.get('latest_snapshot_success_rows_count', 0)}`")
    lines.append(f"- Error rows: `{ocr.get('latest_snapshot_error_rows_count', 0)}`")
    lines.append("")
    lines.append("Per-model metrics (latest snapshot):")
    lines.append("")
    for row in ocr.get("models", []):
        lines.append(
            "- "
            f"{row.get('model_name')}: "
            f"CER={row.get('avg_cer_pct')}, "
            f"WER={row.get('avg_wer_pct')}, "
            f"Layout={row.get('avg_layout_accuracy_pct')}, "
            f"Time={row.get('avg_time_seconds')}s, "
            f"Cost=${row.get('total_cost_usd')}"
        )
    lines.append("")
    lines.append(f"- OCR overall winner: `{ocr.get('verdict', {}).get('overall_winner')}`")
    lines.append("")

    lines.append("## LLM Parsing")
    lines.append(f"- All rows: `{llm.get('all_rows_count', 0)}`")
    lines.append(f"- Latest snapshot rows: `{llm.get('latest_snapshot_rows_count', 0)}`")
    lines.append(f"- Successful rows: `{llm.get('latest_snapshot_success_rows_count', 0)}`")
    lines.append(f"- Error rows: `{llm.get('latest_snapshot_error_rows_count', 0)}`")
    if llm.get("errors_by_model"):
        lines.append(f"- Errors by model: `{llm.get('errors_by_model')}`")
    lines.append("")
    lines.append("Per-model metrics (latest snapshot):")
    lines.append("")
    for row in llm.get("models", []):
        lines.append(
            "- "
            f"{row.get('model_name')}: "
            f"Accuracy={row.get('avg_accuracy_pct')}%, "
            f"Time={row.get('avg_time_seconds')}s, "
            f"Cost=${row.get('total_cost_usd')}, "
            f"JSON valid={row.get('json_valid_rate')}%, "
            f"Schema valid={row.get('schema_valid_rate')}%"
        )
    lines.append("")
    lines.append(
        f"- LLM verdict: best_accuracy=`{llm.get('verdict', {}).get('best_accuracy_model')}`, "
        f"fastest=`{llm.get('verdict', {}).get('fastest_model')}`, "
        f"cheapest=`{llm.get('verdict', {}).get('cheapest_model')}`"
    )
    lines.append("")

    lines.append("## Recommended Endpoint Stack")
    stack = rec.get("recommended_stack_for_endpoint", {})
    lines.append(f"- OCR: `{stack.get('ocr_model')}`")
    lines.append(f"- LLM parsing: `{stack.get('llm_parsing_model')}`")
    lines.append(f"- Rule: {stack.get('selection_rule')}")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OCR/LLM benchmark results.")
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Project root that contains ground_truth_database/",
    )
    parser.add_argument(
        "--out-dir",
        default="exports",
        help="Output directory for exported files.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr_db = base_dir / "ground_truth_database" / "ocr_test_results.db"
    llm_db = base_dir / "ground_truth_database" / "llm_parsing_test_results.db"

    if not ocr_db.exists():
        raise FileNotFoundError(f"Missing OCR database: {ocr_db}")
    if not llm_db.exists():
        raise FileNotFoundError(f"Missing LLM database: {llm_db}")

    ocr_all = load_table(ocr_db, "ocr_test_results")
    llm_all = load_table(llm_db, "llm_parsing_test_results")

    ocr_latest = latest_snapshot(ocr_all)
    llm_latest = latest_snapshot(llm_all)

    ocr_all.to_csv(out_dir / "ocr_all_runs.csv", index=False, encoding="utf-8")
    ocr_latest.to_csv(out_dir / "ocr_latest_snapshot.csv", index=False, encoding="utf-8")
    llm_all.to_csv(out_dir / "llm_all_runs.csv", index=False, encoding="utf-8")
    llm_latest.to_csv(out_dir / "llm_latest_snapshot.csv", index=False, encoding="utf-8")

    ocr_summary = summarize_ocr(ocr_latest, ocr_all)
    llm_summary = summarize_llm(llm_latest, llm_all)
    recommendation = build_recommendation(ocr_summary, llm_summary)

    write_json(out_dir / "ocr_summary_latest.json", ocr_summary)
    write_json(out_dir / "llm_summary_latest.json", llm_summary)
    write_json(out_dir / "benchmark_recommendation.json", recommendation)
    write_markdown_report(out_dir / "RESULTS_SUMMARY.md", ocr_summary, llm_summary, recommendation)

    print(f"Export complete in: {out_dir}")
    for name in [
        "ocr_all_runs.csv",
        "ocr_latest_snapshot.csv",
        "ocr_summary_latest.json",
        "llm_all_runs.csv",
        "llm_latest_snapshot.csv",
        "llm_summary_latest.json",
        "benchmark_recommendation.json",
        "RESULTS_SUMMARY.md",
    ]:
        print(f"- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
