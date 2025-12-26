from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    non_null: int
    nulls: int
    null_pct: float
    unique: int
    top_values: list[tuple[str, int]]


@dataclass(frozen=True)
class NumericProfile:
    name: str
    count: int
    mean: float | None
    median: float | None
    std: float | None
    min: float | None
    p05: float | None
    p95: float | None
    max: float | None
    outlier_high_count: int
    outlier_low_count: int


def _safe_float(x: object) -> float | None:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def basic_overview(df: pd.DataFrame) -> dict[str, object]:
    rows, cols = df.shape
    return {
        "rows": int(rows),
        "cols": int(cols),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }


def infer_datetime_columns(df: pd.DataFrame, *, max_parse_cols: int = 10) -> list[str]:
    # Heuristic: only try object columns (strings) and limit attempts.
    candidates: list[str] = [
        c for c in df.columns if str(df[c].dtype) in {"object", "string"}
    ][:max_parse_cols]

    dt_cols: list[str] = []
    for col in candidates:
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            continue
        sample = non_null.astype(str).head(500)
        # Heuristic parsing can trigger noisy warnings when formats vary; suppress here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        success_rate = float(parsed.notna().mean())
        if success_rate >= 0.9:
            dt_cols.append(col)
    return dt_cols


def column_profiles(df: pd.DataFrame) -> list[ColumnProfile]:
    profiles: list[ColumnProfile] = []
    n = len(df)

    for col in df.columns:
        series = df[col]
        nulls = int(series.isna().sum())
        non_null = int(n - nulls)
        null_pct = float(nulls / n) if n else 0.0
        unique = int(series.nunique(dropna=True))

        top_values: list[tuple[str, int]] = []
        try:
            vc = series.astype("string").value_counts(dropna=True).head(5)
            top_values = [(str(k), int(v)) for k, v in vc.items()]
        except Exception:
            top_values = []

        profiles.append(
            ColumnProfile(
                name=str(col),
                dtype=str(series.dtype),
                non_null=non_null,
                nulls=nulls,
                null_pct=null_pct,
                unique=unique,
                top_values=top_values,
            )
        )

    return profiles


def numeric_profiles(df: pd.DataFrame) -> list[NumericProfile]:
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    profiles: list[NumericProfile] = []

    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        s_non_null = s.dropna()
        if s_non_null.empty:
            profiles.append(
                NumericProfile(
                    name=str(col),
                    count=0,
                    mean=None,
                    median=None,
                    std=None,
                    min=None,
                    p05=None,
                    p95=None,
                    max=None,
                    outlier_high_count=0,
                    outlier_low_count=0,
                )
            )
            continue

        q1 = float(s_non_null.quantile(0.25))
        q3 = float(s_non_null.quantile(0.75))
        iqr = q3 - q1
        # Tukey fences
        low_fence = q1 - 1.5 * iqr
        high_fence = q3 + 1.5 * iqr
        outlier_low = int((s_non_null < low_fence).sum())
        outlier_high = int((s_non_null > high_fence).sum())

        profiles.append(
            NumericProfile(
                name=str(col),
                count=int(s_non_null.shape[0]),
                mean=_safe_float(s_non_null.mean()),
                median=_safe_float(s_non_null.median()),
                std=_safe_float(s_non_null.std(ddof=1)),
                min=_safe_float(s_non_null.min()),
                p05=_safe_float(s_non_null.quantile(0.05)),
                p95=_safe_float(s_non_null.quantile(0.95)),
                max=_safe_float(s_non_null.max()),
                outlier_high_count=outlier_high,
                outlier_low_count=outlier_low,
            )
        )

    return profiles


def top_correlations(df: pd.DataFrame, *, top_n: int = 6) -> list[tuple[str, str, float]]:
    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] < 2:
        return []

    corr = num.corr(numeric_only=True)
    pairs: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            v = corr.loc[a, b]
            if pd.isna(v):
                continue
            pairs.append((str(a), str(b), float(v)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    # Filter out near-zero to avoid noise
    pairs = [p for p in pairs if abs(p[2]) >= 0.3]
    return pairs[:top_n]


def detect_target_column(df: pd.DataFrame) -> str | None:
    """Heuristic: pick a likely outcome/label column if present."""
    candidates = [
        "quality",
        "target",
        "label",
        "outcome",
        "y",
        "churn",
        "converted",
        "conversion",
    ]
    lower = {str(c).lower(): str(c) for c in df.columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return None


def feature_target_correlations(
    df: pd.DataFrame,
    *,
    target_col: str,
    top_n: int = 6,
) -> list[tuple[str, float]]:
    """Return features most associated with a numeric target via correlation."""
    if target_col not in df.columns:
        return []
    if str(df[target_col].dtype) not in {"int64", "int32", "float64", "float32"}:
        # Try coercion
        target = pd.to_numeric(df[target_col], errors="coerce")
    else:
        target = df[target_col]

    if target.dropna().nunique() < 2:
        return []

    num = df.select_dtypes(include=["number"]).copy()
    if target_col not in num.columns:
        num[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    num = num.dropna(subset=[target_col])
    if num.shape[0] < 10 or num.shape[1] < 2:
        return []

    corr = num.corr(numeric_only=True)[target_col].drop(labels=[target_col], errors="ignore")
    corr = corr.dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
    # Ignore tiny correlations (noise)
    corr = corr[corr.abs() >= 0.15]
    return [(str(k), float(v)) for k, v in corr.head(top_n).items()]


def distribution_insights(df: pd.DataFrame, *, top_n: int = 6) -> list[str]:
    """Lightweight, general 'pattern' signals even without a time column."""
    notes: list[str] = []
    num = df.select_dtypes(include=["number"]).copy()
    if num.empty:
        return notes

    # Skewness highlights
    sk = num.skew(numeric_only=True).dropna()
    if not sk.empty:
        sk_sorted = sk.reindex(sk.abs().sort_values(ascending=False).index)
        for col in sk_sorted.head(top_n).index:
            val = float(sk_sorted[col])
            if abs(val) < 1.0:
                continue
            direction = "right-skewed" if val > 0 else "left-skewed"
            notes.append(f"'{col}' is {direction} (a small number of records pull the values to one side).")

    # Outlier rate highlights
    prof = numeric_profiles(df)
    rates: list[tuple[str, float]] = []
    for p in prof:
        if p.count <= 0:
            continue
        rate = float((p.outlier_low_count + p.outlier_high_count) / p.count)
        rates.append((p.name, rate))
    rates.sort(key=lambda x: x[1], reverse=True)
    for name, rate in rates[:top_n]:
        if rate >= 0.02:
            notes.append(f"'{name}' has an elevated outlier rate (~{rate*100:.1f}% of rows flagged by a standard outlier rule).")

    return notes


def potential_id_columns(df: pd.DataFrame) -> list[str]:
    # Heuristic: high uniqueness ratio and not too many nulls.
    n = len(df)
    if n == 0:
        return []

    ids: list[str] = []
    for col in df.columns:
        s = df[col]
        null_pct = float(s.isna().mean())
        if null_pct > 0.2:
            continue
        unique_ratio = float(s.nunique(dropna=True) / n)
        if unique_ratio >= 0.98:
            ids.append(str(col))
    return ids


def suspicious_constant_columns(df: pd.DataFrame) -> list[str]:
    const: list[str] = []
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        if nunique <= 1:
            const.append(str(col))
    return const


def high_missing_columns(df: pd.DataFrame, *, threshold: float = 0.3) -> list[tuple[str, float]]:
    res: list[tuple[str, float]] = []
    for col in df.columns:
        pct = float(df[col].isna().mean())
        if pct >= threshold:
            res.append((str(col), pct))
    res.sort(key=lambda x: x[1], reverse=True)
    return res


def detect_basic_anomalies(df: pd.DataFrame) -> list[str]:
    notes: list[str] = []

    # Duplicate rows
    if len(df) > 0:
        dup = int(df.duplicated().sum())
        if dup > 0:
            notes.append(f"{dup} duplicate rows detected")

    # Negative values in columns that look non-negative
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        if any(k in str(col).lower() for k in ["count", "qty", "quantity", "age", "duration", "days", "minutes", "hours", "revenue", "sales", "price", "amount"]):
            s = pd.to_numeric(df[col], errors="coerce")
            neg = int((s < 0).sum())
            if neg > 0:
                notes.append(f"{neg} negative values in '{col}' (may be invalid depending on meaning)")

    return notes
