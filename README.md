# EDA Report Generator

Generates a plain-English EDA report from a CSV file.

## Quickstart (Windows PowerShell)

1. Put your dataset in `data/` (optional).
2. Run:

```powershell
python -m eda_report --input "data\\sample.csv" --output "reports\\sample_report.md"
```

## Web UI

Run a local website to upload a CSV and view the EDA report with a cleaner UI:

```powershell
pip install -e .
eda-web
```

Common options:

```powershell
eda-web --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

Or via the installed CLI after `pip install -e .`:

```powershell
pip install -e .
eda-report --input "data\\sample.csv" --output "reports\\sample_report.md"
```

## Notes
- The report is Markdown and follows a structured template.
- The tool does not guess business context; it only summarizes what is supported by the data.
