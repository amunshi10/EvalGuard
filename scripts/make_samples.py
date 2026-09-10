"""Build the bundled sample datasets.

The two identity-document samples come from the id-document-classifier project
(ResNet50 features over MIDV-500 crops). This script imports that project's own
`train.py` and `splits.py` so the predictions in the CSVs are produced by the same
linear probe, trained the same way, on the same 2048-dimensional features - not by
an approximation of it.

Two things are reduced so the result can live in a git repository:

  * the feature columns written to CSV are PCA projections of the 2048-dim vectors.
    The similarity structure that the leakage check relies on survives this; frames
    from the same video clip stay near-identical.
  * predictions are computed on the full 2048-dim features before reduction, so the
    accuracy in each CSV matches the number in the source project's own reports.

Usage:
    python scripts/make_samples.py --source /path/to/id-document-classifier
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"

N_COMPONENTS = 64
SEED = 7

# Measured in the source project (reports/results_crop_*.txt); used only to sanity
# check that this script reproduced the same experiment.
EXPECTED = {"random": 0.9725, "doctype": 0.8257}

# reports/stability.txt, sections A and B.
SEED_SCORES = "0.8171, 0.8143, 0.8100, 0.8129, 0.8171"
SPLIT_SCORES = "0.5714, 0.8529, 0.6557, 0.5843, 0.7243"


def import_source_module(src_dir: Path, name: str):
    """Import a module from the classifier project by path."""
    spec = importlib.util.spec_from_file_location(name, src_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_id_sample(source: Path, split_mode: str) -> tuple[pd.DataFrame, dict]:
    src_dir = source / "src"
    # splits.py must be importable first - train.py imports from it by bare name.
    import_source_module(src_dir, "splits")
    train_mod = import_source_module(src_dir, "train")

    data = train_mod.load_features("crop")
    splits = json.loads((source / "data" / "processed" / "splits.json").read_text())
    split_of = {t: s for s, ts in splits.items() for t in ts}
    task_classes = train_mod.TASK_CLASSES

    keep = np.array(
        [
            lbl in task_classes and dt in split_of
            for lbl, dt in zip(data["label"], data["doc_type"])
        ]
    )
    data = {k: v[keep] for k, v in data.items()}

    y = np.array([task_classes.index(l) for l in data["label"]], dtype=np.int64)
    where = train_mod.assign_split(data, split_of, split_mode)
    tr, va, te = where == "train", where == "val", where == "test"
    print(f"  {split_mode}: train={tr.sum()} val={va.sum()} test={te.sum()}")

    head = train_mod.train_head(
        data["feats"][tr], y[tr], data["feats"][va], y[va], len(task_classes)
    )
    head.eval()
    with torch.inference_mode():
        pred_idx = head(torch.from_numpy(data["feats"])).argmax(1).numpy()
    prediction = np.array([task_classes[i] for i in pred_idx])

    accuracy = float((pred_idx[te] == y[te]).mean())
    expected = EXPECTED[split_mode]
    flag = "ok" if abs(accuracy - expected) < 0.02 else "DIFFERS from report"
    print(f"  test accuracy {accuracy:.4f} (project reports {expected:.4f}) - {flag}")

    pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
    reduced = pca.fit_transform(data["feats"])
    print(f"  PCA to {N_COMPONENTS} dims keeps {pca.explained_variance_ratio_.sum():.1%} of variance")

    out = pd.DataFrame(
        {
            "id": data["path"],
            "split": where,
            "label": data["label"],
            "prediction": prediction,
            "doc_type": data["doc_type"],
            "clip": data["clip"],
            "capture_condition": data["condition"],
        }
    )
    for i in range(reduced.shape[1]):
        out[f"f{i:02d}"] = np.round(reduced[:, i], 4)

    return out, {"accuracy": accuracy, "rows": len(out), "test_rows": int(te.sum())}


def build_fraud() -> tuple[pd.DataFrame, dict]:
    """A synthetic, deliberately imbalanced dataset for the baseline check."""
    rng = np.random.default_rng(SEED)
    n = 3000
    label = np.where(rng.random(n) < 0.94, "legitimate", "fraud")
    signal = (label == "fraud").astype(float)

    df = pd.DataFrame(
        {
            "id": [f"txn_{i:05d}" for i in range(n)],
            "split": np.where(rng.random(n) < 0.7, "train", "test"),
            "label": label,
            "amount": np.round(rng.lognormal(3.2 + 0.25 * signal, 1.0, n), 2),
            "hour": rng.integers(0, 24, n),
            "n_prior_txns": rng.poisson(12, n),
            "distance_km": np.round(rng.exponential(30 + 20 * signal, n), 1),
            "device_age_days": rng.integers(0, 1500, n),
        }
    )
    # A near-majority classifier: almost always legitimate, rarely right on fraud.
    caught = (df["label"] == "fraud") & (rng.random(n) < 0.08)
    df["prediction"] = np.where(caught, "fraud", "legitimate")

    test = df[df["split"] == "test"]
    acc = float((test["label"] == test["prediction"]).mean())
    base = float(test["label"].value_counts(normalize=True).iloc[0])
    print(f"  fraud: {len(df):,} rows, test accuracy {acc:.1%}, baseline {base:.1%}")
    return df, {"accuracy": acc, "baseline": base}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "Downloads" / "id-document-classifier",
        help="Path to the id-document-classifier project.",
    )
    args = parser.parse_args()

    SAMPLES_DIR.mkdir(exist_ok=True)
    index: list[dict] = []

    if (args.source / "models" / "features_crop.npz").exists():
        print(f"Reading real features from {args.source}")

        random_df, random_stats = build_id_sample(args.source, "random")
        random_df.to_csv(SAMPLES_DIR / "id_documents_random_split.csv", index=False)
        index.append(
            {
                "id": "id_random",
                "file": "id_documents_random_split.csv",
                "name": "ID documents - random frame split",
                "blurb": (
                    f"Real ResNet50 features from {random_stats['rows']:,} MIDV-500 video frames, "
                    f"split by shuffling frames. Scores {random_stats['accuracy']:.1%} - and "
                    "EvalGuard shows why that number is not real."
                ),
                "expect": "fail",
                "group_col": "doc_type",
                "ignore_cols": "clip, capture_condition",
                "seed_scores": SEED_SCORES,
                "split_scores": SPLIT_SCORES,
            }
        )

        grouped_df, grouped_stats = build_id_sample(args.source, "doctype")
        grouped_df.to_csv(SAMPLES_DIR / "id_documents_grouped_split.csv", index=False)
        index.append(
            {
                "id": "id_grouped",
                "file": "id_documents_grouped_split.csv",
                "name": "ID documents - document-design split",
                "blurb": (
                    "The same frames, re-split so entire document designs are held out. The "
                    f"leakage is gone and accuracy falls to {grouped_stats['accuracy']:.1%} - but "
                    "the variance check shows even that number swings by 11 points depending on "
                    "which designs you hold out."
                ),
                "expect": "fail",
                "group_col": "doc_type",
                "ignore_cols": "clip, capture_condition",
                "seed_scores": SEED_SCORES,
                "split_scores": SPLIT_SCORES,
            }
        )
    else:
        print(f"! {args.source} has no cached features - skipping identity-document samples.")

    fraud_df, fraud_stats = build_fraud()
    fraud_df.to_csv(SAMPLES_DIR / "fraud_imbalanced.csv", index=False)
    index.append(
        {
            "id": "fraud",
            "file": "fraud_imbalanced.csv",
            "name": "Fraud detection - imbalanced classes",
            "blurb": (
                f"Synthetic. {fraud_stats['accuracy']:.1%} accuracy sounds strong until you notice "
                f"that always guessing 'legitimate' scores {fraud_stats['baseline']:.1%}."
            ),
            "expect": "fail",
            "group_col": "",
            "ignore_cols": "",
            "seed_scores": "",
            "split_scores": "",
        }
    )

    (SAMPLES_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nWrote {len(index)} samples to {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
