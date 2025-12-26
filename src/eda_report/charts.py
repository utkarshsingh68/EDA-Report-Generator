from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Headless-friendly backend (important on servers/CI)
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass(frozen=True)
class ChartArtifact:
    title: str
    filename: str


def _safe_stem(name: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in name.strip())
    stem = stem.strip("-")
    return stem or "dataset"


def generate_basic_charts(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    dataset_name: str,
    target_col: str | None = None,
) -> list[ChartArtifact]:
    """Generate a small set of commonly useful charts.

    Produces PNG files inside out_dir and returns metadata for embedding.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(Path(dataset_name).stem)

    artifacts: list[ChartArtifact] = []

    # 1) Target distribution (pie if categorical-ish, else histogram)
    if target_col and target_col in df.columns:
        s = df[target_col].dropna()
        nunique = int(s.nunique())
        if nunique > 0:
            plt.figure(figsize=(7, 4.2))
            if nunique <= 10:
                counts = s.value_counts().sort_index()
                plt.pie(
                    counts.values,
                    labels=[str(x) for x in counts.index],
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"fontsize": 9},
                )
                plt.title(f"{target_col} distribution")
            else:
                sns.histplot(pd.to_numeric(s, errors="coerce"), bins=20, kde=False)
                plt.title(f"{target_col} distribution")
                plt.xlabel(target_col)
                plt.ylabel("Count")

            fname = f"{stem}-target.png"
            path = out_dir / fname
            plt.tight_layout()
            plt.savefig(path, dpi=160)
            plt.close()
            artifacts.append(ChartArtifact(title=f"{target_col} distribution", filename=fname))

    # 2) Correlation heatmap (numeric columns)
    num = df.select_dtypes(include=["number"]).copy()
    if num.shape[1] >= 2:
        # Keep it readable by limiting to top-variance columns
        variances = num.var(numeric_only=True).sort_values(ascending=False)
        cols = list(variances.head(12).index)
        corr = num[cols].corr(numeric_only=True)

        plt.figure(figsize=(8.5, 6.5))
        sns.heatmap(
            corr,
            vmin=-1,
            vmax=1,
            cmap="vlag",
            center=0,
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Correlation heatmap (top numeric columns)")
        fname = f"{stem}-corr.png"
        path = out_dir / fname
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        artifacts.append(ChartArtifact(title="Correlation heatmap", filename=fname))

    # 3) Target vs strongest numeric driver (binned mean)
    if target_col and target_col in df.columns:
        target = pd.to_numeric(df[target_col], errors="coerce")
        num = df.select_dtypes(include=["number"]).copy()
        if target_col not in num.columns:
            num[target_col] = target

        if num.shape[1] >= 2 and target.dropna().nunique() >= 2:
            corr = num.corr(numeric_only=True)[target_col].drop(labels=[target_col], errors="ignore")
            corr = corr.dropna()
            if not corr.empty:
                feature = str(corr.abs().sort_values(ascending=False).index[0])
                feature_s = pd.to_numeric(df[feature], errors="coerce")
                tmp = pd.DataFrame({"feature": feature_s, "target": target}).dropna()
                if tmp.shape[0] >= 30 and tmp["feature"].nunique() >= 8:
                    # Use quantile bins for stability.
                    try:
                        tmp["bin"] = pd.qcut(tmp["feature"], q=8, duplicates="drop")
                    except Exception:
                        tmp["bin"] = pd.cut(tmp["feature"], bins=8)

                    grouped = (
                        tmp.groupby("bin", observed=True)
                        .agg(mean_target=("target", "mean"), n=("target", "size"))
                        .reset_index()
                    )

                    plt.figure(figsize=(8.5, 4.2))
                    sns.barplot(data=grouped, x="bin", y="mean_target", color="#7aa2ff")
                    plt.xticks(rotation=30, ha="right")
                    plt.title(f"Average {target_col} by {feature} (binned)")
                    plt.xlabel(feature)
                    plt.ylabel(f"Average {target_col}")

                    fname = f"{stem}-{target_col}-by-{_safe_stem(feature)}.png"
                    path = out_dir / fname
                    plt.tight_layout()
                    plt.savefig(path, dpi=160)
                    plt.close()
                    artifacts.append(ChartArtifact(title=f"Average {target_col} by {feature} (binned)", filename=fname))

    return artifacts
