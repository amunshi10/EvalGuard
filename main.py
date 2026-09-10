"""EvalGuard web app.

Runs on 0.0.0.0:5000 so Replit's webview picks it up without extra configuration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from evalguard.checks import leakage
from evalguard.loader import DatasetError, load_dataset
from evalguard.report import analyse
from evalguard.checks.variance import parse_scores

APP_ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = APP_ROOT / "samples"
MAX_UPLOAD_BYTES = 32 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def load_samples() -> list[dict]:
    """Sample datasets are described by samples/index.json."""
    index = SAMPLES_DIR / "index.json"
    if not index.exists():
        return []
    with index.open(encoding="utf-8") as fh:
        return json.load(fh)


@app.route("/")
def index():
    return render_template("index.html", samples=load_samples())


@app.route("/api/samples/<name>")
def sample_meta(name: str):
    for sample in load_samples():
        if sample["id"] == name:
            return jsonify(sample)
    return jsonify({"error": "Unknown sample."}), 404


@app.route("/api/analyze", methods=["POST"])
def analyze():
    threshold = request.form.get("threshold", type=float) or leakage.DEFAULT_NEAR_THRESHOLD
    threshold = min(max(threshold, 0.5), 1.0)
    seed_scores = parse_scores(request.form.get("seed_scores", ""))
    split_scores = parse_scores(request.form.get("split_scores", ""))

    source = None
    sample_id = request.form.get("sample")
    if sample_id:
        match = next((s for s in load_samples() if s["id"] == sample_id), None)
        if match is None:
            return jsonify({"error": "Unknown sample dataset."}), 400
        candidate = (SAMPLES_DIR / match["file"]).resolve()
        # Guard against a crafted sample id escaping the samples directory.
        if not str(candidate).startswith(str(SAMPLES_DIR.resolve())) or not candidate.exists():
            return jsonify({"error": "Sample file is missing."}), 400
        source = candidate
    else:
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return jsonify({"error": "Choose a CSV file or pick a sample dataset."}), 400
        source = upload.stream

    group_override = (request.form.get("group_col") or "").strip() or None
    ignore_cols = [
        part.strip()
        for part in (request.form.get("ignore_cols") or "").split(",")
        if part.strip()
    ]

    try:
        dataset = load_dataset(source, group_override=group_override, ignore_cols=ignore_cols)
    except DatasetError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Could not read that file: {exc}"}), 400

    try:
        report = analyse(
            dataset,
            seed_scores=seed_scores,
            split_scores=split_scores,
            similarity_threshold=threshold,
        )
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"Checks failed to run: {exc}"}), 500

    return jsonify(report)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
