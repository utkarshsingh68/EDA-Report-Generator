from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime
import io
import json
import zipfile

from flask import Flask, flash, redirect, render_template, request, url_for
from flask import send_from_directory
from flask import send_file
from markdown import markdown
from werkzeug.utils import secure_filename

from .analyze import detect_target_column
from .charts import generate_basic_charts
from .io import read_table
from .report import generate_report_markdown


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("EDA_REPORT_SECRET") or os.urandom(16)
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("EDA_REPORT_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))

    base_dir = Path(os.environ.get("EDA_REPORT_BASE_DIR", "")) if os.environ.get("EDA_REPORT_BASE_DIR") else Path.cwd()
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    uploads_dir = reports_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/charts/<path:filename>")
    def charts(filename: str):
        charts_dir = reports_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        return send_from_directory(charts_dir, filename)

    @app.get("/download/<path:report_file>")
    def download(report_file: str):
        # Package the report markdown and its chart images into a zip.
        report_path = reports_dir / Path(report_file).name
        if not report_path.exists():
            return ("Report not found", 404)

        meta_path = report_path.with_suffix(report_path.suffix + ".json")
        chart_files: list[str] = []
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                chart_files = [str(x) for x in meta.get("charts", [])]
            except Exception:
                chart_files = []

        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(report_path.name, report_path.read_text(encoding="utf-8"))
            charts_dir = reports_dir / "charts"
            for fname in chart_files:
                p = charts_dir / Path(fname).name
                if p.exists():
                    z.write(p, arcname=str(Path("charts") / p.name))
        mem.seek(0)

        zip_name = f"{report_path.stem}.zip"
        return send_file(mem, mimetype="application/zip", as_attachment=True, download_name=zip_name)

    @app.post("/report")
    def report():
        uploaded = request.files.get("file")

        # Upload-only flow.
        if uploaded is None or uploaded.filename == "":
            flash("Please upload a CSV file.", "error")
            return redirect(url_for("index"))
        if not uploaded.filename.lower().endswith(".csv"):
            flash("Only .csv files are supported.", "error")
            return redirect(url_for("index"))

        # Save to reports/uploads for reproducibility.
        safe_name = secure_filename(Path(uploaded.filename).name) or "upload.csv"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dataset_path = uploads_dir / f"{timestamp}-{safe_name}"
        uploaded.save(dataset_path)
        df = read_table(dataset_path, max_rows=200_000)
        dataset_name = safe_name

        charts_dir = reports_dir / "charts"
        target = detect_target_column(df)
        artifacts = generate_basic_charts(df, out_dir=charts_dir, dataset_name=dataset_name, target_col=target)
        chart_links = [(a.title, f"charts/{a.filename}") for a in artifacts]

        report_md = generate_report_markdown(df, dataset_name=dataset_name, chart_files=chart_links)
        report_html = markdown(report_md, extensions=["tables", "fenced_code"])

        # Persist the markdown so users can share or archive it.
        report_name = f"{Path(dataset_name).stem}-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        report_path = reports_dir / report_name
        report_path.write_text(report_md, encoding="utf-8")

        # Sidecar metadata for download packaging.
        (report_path.with_suffix(report_path.suffix + ".json")).write_text(
            json.dumps({"charts": [Path(p).name for _, p in chart_links]}, indent=2),
            encoding="utf-8",
        )

        return render_template(
            "report.html",
            dataset_name=dataset_name,
            report_md=report_md,
            report_html=report_html,
            report_file=report_name,
            charts=[{"title": t, "url": url_for("charts", filename=Path(p).name)} for t, p in chart_links],
            download_url=url_for("download", report_file=report_name),
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the EDA Report web UI")
    parser.add_argument("--host", default=os.environ.get("EDA_REPORT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("EDA_REPORT_PORT", "8000")))
    parser.add_argument("--debug", action="store_true", default=bool(os.environ.get("EDA_REPORT_DEBUG")))
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
