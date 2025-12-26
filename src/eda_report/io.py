from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def _sniff_csv_dialect(text_sample: str) -> csv.Dialect | None:
    try:
        sniffer = csv.Sniffer()
        return sniffer.sniff(text_sample, delimiters=[",", ";", "\t", "|"])
    except Exception:
        return None


def read_table(path: Path, *, max_rows: int) -> pd.DataFrame:
    """Read a CSV into a DataFrame with basic guardrails."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Try utf-8 first, then fallback to latin-1.
    # Also detect delimiter so common variants (e.g., ';' in some UCI datasets) parse correctly.
    read_kwargs = dict(low_memory=False)

    raw: str
    encoding_used = "utf-8"
    try:
        raw = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        encoding_used = "latin-1"
        raw = path.read_text(encoding="latin-1", errors="replace")

    # Use only a small slice for sniffing.
    sample = raw[:50_000]
    dialect = _sniff_csv_dialect(sample)

    if dialect is not None:
        df = pd.read_csv(
            path,
            encoding=encoding_used,
            nrows=max_rows,
            sep=dialect.delimiter,
            quotechar=getattr(dialect, "quotechar", '"'),
            **read_kwargs,
        )
    else:
        df = pd.read_csv(path, encoding=encoding_used, nrows=max_rows, **read_kwargs)

    return df
