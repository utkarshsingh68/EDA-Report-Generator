# EDA Report Generator

Generates a plain-English EDA report from a CSV file.

## Run Locally (Windows PowerShell)

```powershell
git clone https://github.com/utkarshsingh68/EDA-Report-Generator.git
Set-Location .\EDA-Report-Generator

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e .
```

### Start the website

```powershell
eda-web
```

Open `http://127.0.0.1:8000` and upload a `.csv`.

### Generate a report from the CLI

```powershell
eda-report --input "path\to\data.csv" --output "reports\data-report.md"
```

The CLI also generates charts in `reports/charts/` and embeds them into the Markdown report.

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
