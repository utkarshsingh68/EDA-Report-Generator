from __future__ import annotations

import math
from datetime import date

import pandas as pd

from .analyze import (
    basic_overview,
    column_profiles,
    detect_basic_anomalies,
    detect_target_column,
    distribution_insights,
    feature_target_correlations,
    high_missing_columns,
    infer_datetime_columns,
    numeric_profiles,
    potential_id_columns,
    suspicious_constant_columns,
    top_correlations,
)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_num(x: float | None) -> str:
    if x is None:
        return "n/a"
    if math.isfinite(x) and abs(x) >= 1000:
        return f"{x:,.2f}"
    return f"{x:.3g}" if math.isfinite(x) else "n/a"


def generate_report_markdown(
    df: pd.DataFrame,
    *,
    dataset_name: str = "dataset",
    chart_files: list[tuple[str, str]] | None = None,
) -> str:
    today = date.today().isoformat()

    overview = basic_overview(df)
    cols = column_profiles(df)
    nums = numeric_profiles(df)
    corr = top_correlations(df)

    dt_cols = infer_datetime_columns(df)
    ids = potential_id_columns(df)
    const_cols = suspicious_constant_columns(df)
    missing_hi = high_missing_columns(df)
    anomalies = detect_basic_anomalies(df)

    # --- Build narrative bullets (data-driven, non-prescriptive) ---
    key_insights: list[str] = []

    if missing_hi:
        col, pct = missing_hi[0]
        key_insights.append(
            f"Some fields have substantial missing data (highest: '{col}' at {_fmt_pct(pct)} missing)."
        )

    if corr:
        a, b, v = corr[0]
        strength = "strong" if abs(v) >= 0.7 else "moderate"
        direction = "positive" if v > 0 else "negative"
        key_insights.append(
            f"There is a {strength} {direction} relationship between '{a}' and '{b}' (correlation {v:.2f})."
        )

    if ids:
        key_insights.append(
            f"Likely identifier columns detected (nearly all values unique): {', '.join(ids[:5])}."
        )

    if not key_insights:
        key_insights.append("No single dominant pattern stood out from a high-level scan; the dataset looks broadly well-formed.")

    trends: list[str] = []
    if dt_cols:
        trends.append(
            f"Date-like columns were detected ({', '.join(dt_cols[:3])}); time-based trends may be available once a primary date is chosen."
        )

    # Target-driven patterns (common for many datasets)
    target = detect_target_column(df)
    if target is not None:
        ft = feature_target_correlations(df, target_col=target)
        if ft:
            top_pos = [(c, v) for c, v in ft if v > 0][:3]
            top_neg = [(c, v) for c, v in ft if v < 0][:3]
            if top_pos:
                shown = ", ".join([f"{c} (r={v:.2f})" for c, v in top_pos])
                trends.append(f"Higher '{target}' tends to come with higher: {shown}.")
            if top_neg:
                shown = ", ".join([f"{c} (r={v:.2f})" for c, v in top_neg])
                trends.append(f"Higher '{target}' tends to come with lower: {shown}.")
        else:
            trends.append(f"A likely outcome column '{target}' was detected, but there wasn’t enough signal to summarize strong relationships automatically.")

    # Distribution-based patterns (works even without time)
    trends.extend(distribution_insights(df)[:6])

    if not trends:
        trends.append("No clear time trend was inferred automatically; consider specifying which column represents time.")

    data_quality: list[str] = []
    if const_cols:
        data_quality.append(
            f"Some columns are constant (only one unique value): {', '.join(const_cols[:5])}."
        )
    if missing_hi:
        shown = ", ".join([f"{c} ({_fmt_pct(p)})" for c, p in missing_hi[:5]])
        data_quality.append(f"High-missingness columns (>=30% missing): {shown}.")
    if anomalies:
        data_quality.extend(anomalies)
    if not data_quality:
        data_quality.append("No major data quality issues were detected by basic checks (missingness, duplicates, simple validity heuristics).")

    # Plain English explanation is careful: observations vs assumptions.
    plain = [
        "Observations are based only on what is present in the file.",
        "Where relationships appear (like correlations), they describe variables moving together but do not prove one causes the other.",
    ]

    implications: list[str] = []
    if corr:
        implications.append(
            "If you track both correlated metrics, they may be partially redundant; you might focus reporting on the one that best reflects your business goal."
        )
    if missing_hi:
        implications.append(
            "Fields with heavy missing data can bias results; you may need to confirm whether missing values are expected (e.g., not applicable) or a capture problem."
        )
    if dt_cols:
        implications.append(
            "If decisions depend on changes over time, choosing a single primary date column and aggregating by week/month is usually the next step."
        )
    if not implications:
        implications.append(
            "The dataset appears suitable for deeper analysis once you define the key outcome metric and the main segmentation dimensions."
        )

    limitations = [
        "This report uses automated heuristics (e.g., to detect date columns and outliers); domain meaning may differ.",
        "If the file is a truncated extract or missing historical periods, trend conclusions will be limited.",
    ]

    next_qs: list[str] = [
        "Which column represents the primary business outcome (e.g., conversions, churn, revenue)?",
        "Which column should be treated as the 'time' field for trends?",
        "Are missing values truly unknown, or do they mean 'not applicable'?",
    ]

    # --- Render exactly in the required section headings ---
    lines: list[str] = []

    lines.append("Dataset Overview:")
    lines.append(
        f"- Name: {dataset_name}; Generated: {today}; Shape: {overview['rows']:,} rows × {overview['cols']:,} columns; Memory: {overview['memory_mb']:.1f} MB"
    )

    lines.append("Key Insights:")
    for s in key_insights[:5]:
        lines.append(f"- {s}")

    lines.append("Notable Trends & Patterns:")
    for s in trends[:6]:
        lines.append(f"- {s}")

    lines.append("Anomalies & Data Quality Issues:")
    for s in data_quality[:8]:
        lines.append(f"- {s}")

    lines.append("Plain-English Explanation:")
    for s in plain:
        lines.append(f"- {s}")

    lines.append("Potential Implications:")
    for s in implications[:6]:
        lines.append(f"- {s}")

    lines.append("Limitations & Assumptions:")
    for s in limitations:
        lines.append(f"- {s}")

    lines.append("Next Questions to Explore:")
    for s in next_qs:
        lines.append(f"- {s}")

    # Add compact appendix for transparency (kept short)
    lines.append("")
    lines.append("---")
    lines.append("Appendix (compact stats):")

    if chart_files:
        lines.append("Charts:")
        for title, rel_path in chart_files[:6]:
            lines.append(f"- {title}: ![]({rel_path})")

    lines.append("Columns:")
    for cp in cols[:20]:
        lines.append(
            f"- {cp.name}: type={cp.dtype}, missing={_fmt_pct(cp.null_pct)}, unique={cp.unique}"
        )

    if nums:
        lines.append("Numeric summary (first 10):")
        for p in nums[:10]:
            lines.append(
                f"- {p.name}: count={p.count}, median={_fmt_num(p.median)}, p05={_fmt_num(p.p05)}, p95={_fmt_num(p.p95)}, outliers(low/high)={p.outlier_low_count}/{p.outlier_high_count}"
            )

    if corr:
        lines.append("Top correlations (|r|>=0.3):")
        for a, b, v in corr:
            lines.append(f"- {a} vs {b}: r={v:.2f}")

    return "\n".join(lines) + "\n"
