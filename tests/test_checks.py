"""Tests for the four checks.

Each test builds the smallest dataset that exhibits one fault, so a failure points
at a specific behaviour rather than at "something in the pipeline changed".
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from evalguard.checks import Status
from evalguard.checks import baseline, integrity, leakage, variance
from evalguard.loader import DatasetError, load_dataset


def make_csv(df: pd.DataFrame) -> io.StringIO:
    return io.StringIO(df.to_csv(index=False))


def synthetic(n_per_class: int = 60, n_features: int = 32, seed: int = 0) -> pd.DataFrame:
    """Two well-separated classes in a space big enough for similarity to mean something."""
    rng = np.random.default_rng(seed)
    rows = []
    for cls in ("a", "b"):
        centre = rng.normal(0, 1, n_features) * (1 if cls == "a" else -1)
        for i in range(n_per_class):
            features = centre + rng.normal(0, 0.35, n_features)
            rows.append({"id": f"{cls}{i}", "label": cls, **{f"f{j}": v for j, v in enumerate(features)}})
    df = pd.DataFrame(rows)
    df["split"] = np.where(rng.random(len(df)) < 0.6, "train", "test")
    return df


# ---------------------------------------------------------------- loader

def test_loader_requires_split_and_label():
    with pytest.raises(DatasetError, match="split"):
        load_dataset(make_csv(pd.DataFrame({"label": ["a"], "f0": [1]})))
    with pytest.raises(DatasetError, match="label"):
        load_dataset(make_csv(pd.DataFrame({"split": ["train"], "f0": [1]})))


def test_loader_normalises_split_aliases():
    df = pd.DataFrame(
        {"partition": ["Training", "TEST", "validation"], "target": ["a", "b", "a"], "f0": [1, 2, 3]}
    )
    ds = load_dataset(make_csv(df))
    assert set(ds.df[ds.split_col]) == {"train", "test", "val"}
    assert ds.holdout_split == "test"


def test_loader_honours_group_override_and_ignore():
    df = synthetic(20, 8)
    df["batch"] = "b1"
    ds = load_dataset(make_csv(df), group_override="batch", ignore_cols=["f0", "f1"])
    assert ds.group_col == "batch"
    assert "f0" not in ds.feature_cols and "f1" not in ds.feature_cols


# ---------------------------------------------------------------- leakage

def test_leakage_flags_duplicated_rows():
    df = synthetic()
    train = df[df["split"] == "train"]
    # Copy 20 training rows into the test set under new ids - textbook leakage.
    leaked = train.head(20).copy()
    leaked["split"] = "test"
    leaked["id"] = leaked["id"] + "_copy"
    ds = load_dataset(make_csv(pd.concat([df, leaked], ignore_index=True)))

    result = leakage.run(ds)
    assert result.status == Status.FAIL
    assert result.metrics["Near-duplicates found"] != "0"


def test_leakage_passes_on_clean_split():
    ds = load_dataset(make_csv(synthetic()))
    assert leakage.run(ds).status == Status.PASS


def test_leakage_skips_low_dimensional_features():
    """Cosine similarity is not informative in a handful of dimensions."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(400)],
            "label": rng.choice(["a", "b"], 400),
            "split": rng.choice(["train", "test"], 400),
            "x": rng.normal(size=400),
            "y": rng.normal(size=400),
            "z": rng.normal(size=400),
        }
    )
    result = leakage.run(load_dataset(make_csv(df)))
    assert result.status == Status.SKIP
    assert "too few" in result.headline


# ---------------------------------------------------------------- baseline

def test_baseline_fails_when_model_barely_beats_majority():
    n = 500
    rng = np.random.default_rng(2)
    label = np.where(rng.random(n) < 0.93, "no", "yes")
    df = pd.DataFrame(
        {
            "id": range(n),
            "split": "test",
            "label": label,
            "prediction": "no",  # constant predictor
            "f0": rng.normal(size=n),
        }
    )
    result = baseline.run(load_dataset(make_csv(df)))
    assert result.status == Status.FAIL
    assert "94" in result.metrics["Always-guess-majority accuracy"] or "93" in result.metrics[
        "Always-guess-majority accuracy"
    ]


def test_baseline_passes_with_real_gain():
    n = 400
    rng = np.random.default_rng(3)
    label = rng.choice(["a", "b"], n)
    # 90% correct on a balanced problem is a genuine 40-point gain.
    prediction = np.where(rng.random(n) < 0.9, label, np.where(label == "a", "b", "a"))
    df = pd.DataFrame(
        {"id": range(n), "split": "test", "label": label, "prediction": prediction,
         "f0": rng.normal(size=n)}
    )
    assert baseline.run(load_dataset(make_csv(df))).status == Status.PASS


def test_baseline_reports_without_predictions():
    df = synthetic(40, 8).drop(columns=[])
    result = baseline.run(load_dataset(make_csv(df)))
    assert result.status in (Status.PASS, Status.WARN)
    assert "Always-guess-majority accuracy" in result.metrics


# ---------------------------------------------------------------- variance

def test_parse_scores_accepts_mixed_formats():
    assert variance.parse_scores("0.81, 0.82") == pytest.approx([0.81, 0.82])
    assert variance.parse_scores("81.5 82.5") == pytest.approx([0.815, 0.825])
    # Seed numbers are not scores.
    assert variance.parse_scores("seed 1000: acc 0.8171") == pytest.approx([0.8171])


def test_variance_flags_split_noise_exceeding_seed_noise():
    result = variance.run(
        seed_scores=[0.8171, 0.8143, 0.8100, 0.8129, 0.8171],
        split_scores=[0.5714, 0.8529, 0.6557, 0.5843, 0.7243],
    )
    assert result.status == Status.FAIL
    assert "times more" in result.headline


def test_variance_passes_when_tight():
    result = variance.run(seed_scores=[], split_scores=[0.900, 0.902, 0.901, 0.9005])
    assert result.status == Status.PASS


def test_variance_skips_without_input():
    assert variance.run([], []).status == Status.SKIP


# ---------------------------------------------------------------- integrity

def test_integrity_flags_group_overlap():
    df = synthetic(60, 16)
    df["doc"] = ["d" + str(i % 12) for i in range(len(df))]  # groups span both splits
    result = integrity.run(load_dataset(make_csv(df), group_override="doc"))
    assert result.status == Status.FAIL
    assert "doc" in result.headline


def test_integrity_flags_shared_ids():
    df = synthetic(40, 16)
    dup = df[df["split"] == "train"].head(5).copy()
    dup["split"] = "test"
    result = integrity.run(load_dataset(make_csv(pd.concat([df, dup], ignore_index=True))))
    assert result.status == Status.FAIL


def test_integrity_flags_class_missing_from_train():
    df = synthetic(40, 16)
    df.loc[df["label"] == "b", "split"] = "test"  # class b never trained on
    result = integrity.run(load_dataset(make_csv(df)))
    assert result.status == Status.FAIL


def test_integrity_passes_on_clean_grouped_split():
    df = synthetic(300, 16)
    # Every row is its own group, so no group can straddle the split.
    df["doc"] = ["d" + str(i) for i in range(len(df))]
    # Split within each class so both sides see the same class mix.
    rank = df.groupby("label").cumcount()
    per_class = df.groupby("label")["label"].transform("size")
    df["split"] = np.where(rank < per_class * 0.6, "train", "test")

    result = integrity.run(load_dataset(make_csv(df), group_override="doc"))
    assert result.status == Status.PASS, result.headline
