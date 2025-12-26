from __future__ import annotations

import argparse
from pathlib import Path

from .analyze import detect_target_column
from .charts import generate_basic_charts
from .io import read_table
from .report import generate_report_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an EDA Markdown report from a CSV file")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output Markdown report")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=200_000,
        help="Max rows to load (safety guard for huge files)",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path, max_rows=args.max_rows)

    # Create charts alongside the report output.
    charts_dir = output_path.parent / "charts"
    target = detect_target_column(df)
    artifacts = generate_basic_charts(df, out_dir=charts_dir, dataset_name=input_path.name, target_col=target)
    chart_links = [(a.title, f"charts/{a.filename}") for a in artifacts]

    report_md = generate_report_markdown(df, dataset_name=input_path.name, chart_files=chart_links)

    output_path.write_text(report_md, encoding="utf-8")
    print(f"Wrote report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
