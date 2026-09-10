"""Check 1 - near-duplicate and leakage detection.

The failure this catches: a sample in the test set is identical, or nearly
identical, to a sample the model already saw during training. The model can
recall it rather than generalise to it, so the reported test accuracy is partly
a memory score. It is the most common way a benchmark number turns out to be
wrong, and it is invisible to every metric computed after the split.

Implementation: features are encoded once (see features.py), then for each pair
of splits a cosine nearest-neighbour lookup finds, for every held-out row, its
most similar training row. Anything at or above the similarity threshold is
reported. Exact duplicates are found separately via row fingerprints, because
"byte-identical row" is a stronger and more legible claim than "cosine 0.999".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from evalguard.checks.base import CheckResult, Status, skipped
from evalguard.features import build_matrix, fingerprint
from evalguard.loader import Dataset

# Above this, two rows are treated as the same sample wearing a different id.
DEFAULT_NEAR_THRESHOLD = 0.95
# Analysing every pair is quadratic; beyond this many rows we subsample and say so.
MAX_ROWS_PER_SPLIT = 8_000
# Contamination above this share of the held-out set is a failure rather than a
# warning. Any exact duplicate across splits fails regardless of the rate.
FAIL_FRACTION = 0.05
MAX_PAIRS_SHOWN = 25
# A pair must also be this unusual relative to ordinary pairs in the same dataset.
BACKGROUND_PERCENTILE = 99.5
BACKGROUND_PAIRS = 40_000
# Accuracy differences smaller than this are not worth putting in the headline.
MIN_REPORTABLE_DROP_PTS = 0.5
# Below this many encoded dimensions, similarity-based duplicate detection is
# not trustworthy - see _low_dimensional_result for why.
MIN_DIMS_FOR_SIMILARITY = 16


def _low_dimensional_result(
    key: str, title: str, n_dims: int, exact_count: int, holdout_rows: int
) -> CheckResult:
    """Report honestly that this feature space is too small to judge similarity in.

    In a handful of dimensions, a few thousand points fill the space densely enough
    that every held-out row has a close nearest neighbour purely by chance. Cosine
    similarity above 0.95 stops meaning "these are the same sample" and starts
    meaning "this space is small". Reporting a contamination rate here would be
    confidently wrong, so the check reports what it can stand behind - exact
    duplicates - and defers the rest to the id and group checks.
    """
    if exact_count:
        return CheckResult(
            key=key,
            title=title,
            status=Status.FAIL,
            headline=(
                f"{exact_count:,} held-out rows are exact duplicates of training rows."
            ),
            why_it_matters=(
                "These rows are byte-identical to samples the model trained on, so getting "
                "them right measures recall rather than generalisation."
            ),
            what_to_do=(
                "Remove the duplicated rows, or re-split so that duplicates travel together."
            ),
            metrics={
                "Held-out rows checked": f"{holdout_rows:,}",
                "Exact duplicates across splits": f"{exact_count:,}",
                "Encoded feature dimensions": str(n_dims),
            },
            notes=[
                f"Only exact duplicates were checked: with {n_dims} feature dimensions, "
                "similarity-based near-duplicate detection is not reliable."
            ],
        )

    return CheckResult(
        key=key,
        title=title,
        status=Status.SKIP,
        headline=(
            f"No exact duplicates found. Near-duplicate detection was skipped because "
            f"{n_dims} feature dimensions is too few to measure similarity meaningfully."
        ),
        why_it_matters=(
            f"With only {n_dims} dimensions, a few thousand rows pack the space densely "
            "enough that every held-out row has a close nearest neighbour by chance. Any "
            "contamination rate computed here would reflect the size of the feature space "
            "rather than genuine duplication, so it is better not to report one."
        ),
        what_to_do=(
            "Rely on the split-integrity check below, which compares ids and groups directly "
            "and does not depend on feature similarity. If your rows are not independent, "
            "make sure a group column is set."
        ),
        metrics={
            "Held-out rows checked": f"{holdout_rows:,}",
            "Exact duplicates across splits": "0",
            "Encoded feature dimensions": str(n_dims),
        },
    )


def _background_similarity(matrix, percentile: float = BACKGROUND_PERCENTILE) -> float | None:
    """How similar are two ordinary, unrelated rows in this dataset?

    A fixed cosine cut-off is meaningless on its own: with five non-negative
    numeric columns almost every pair scores above 0.95, while for sparse text
    features 0.6 is already suspicious. Sampling random pairs gives a baseline for
    what "ordinary" looks like here, so a duplicate has to be an outlier against
    the dataset's own distribution rather than against an arbitrary constant.
    """
    n = matrix.shape[0]
    if n < 32:
        return None
    rng = np.random.default_rng(0)
    left = rng.integers(0, n, BACKGROUND_PAIRS)
    right = rng.integers(0, n, BACKGROUND_PAIRS)
    keep = left != right
    if not keep.any():
        return None
    # Rows are L2-normalised, so an elementwise product summed is the cosine.
    sims = np.asarray(matrix[left[keep]].multiply(matrix[right[keep]]).sum(axis=1)).ravel()
    return float(np.percentile(sims, percentile))


def _pair_rows(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    sims: np.ndarray,
    neighbour_idx: np.ndarray,
    threshold: float,
    dataset: Dataset,
) -> list[dict]:
    """Build the human-readable table of offending pairs, worst first."""
    hits = np.where(sims >= threshold)[0]
    order = hits[np.argsort(-sims[hits])]

    rows = []
    for pos in order[:MAX_PAIRS_SHOWN]:
        holdout_row = holdout_df.iloc[pos]
        train_row = train_df.iloc[neighbour_idx[pos]]
        rows.append(
            {
                "similarity": round(float(sims[pos]), 4),
                "holdout_id": _display_id(holdout_row, holdout_df.index[pos], dataset),
                "train_id": _display_id(train_row, train_df.index[neighbour_idx[pos]], dataset),
                "holdout_label": str(holdout_row[dataset.label_col]),
                "train_label": str(train_row[dataset.label_col]),
            }
        )
    return rows


def _display_id(row: pd.Series, fallback_index, dataset: Dataset) -> str:
    if dataset.id_col:
        return str(row[dataset.id_col])
    return f"row {fallback_index}"


def _accuracy(df: pd.DataFrame, dataset: Dataset) -> float | None:
    if dataset.pred_col is None or df.empty:
        return None
    correct = df[dataset.label_col].astype(str) == df[dataset.pred_col].astype(str)
    return float(correct.mean())


def run(dataset: Dataset, threshold: float = DEFAULT_NEAR_THRESHOLD) -> CheckResult:
    key, title = "leakage", "Train/test leakage"

    if not dataset.feature_cols:
        return skipped(
            key,
            title,
            "No feature columns, so there is nothing to compare samples on.",
            "Include the feature values you trained on (or an embedding, one column "
            "per dimension) alongside id, split and label.",
        )

    holdout_split = dataset.holdout_split
    if holdout_split is None or "train" not in dataset.splits:
        return skipped(
            key,
            title,
            "Needs both a train split and a held-out split to compare.",
            "Check that the split column contains train and test values.",
        )

    df = dataset.df
    notes: list[str] = []

    # Subsample large splits so the nearest-neighbour search stays interactive.
    frames = {}
    for split in ("train", holdout_split):
        rows = dataset.rows_for(split)
        if len(rows) > MAX_ROWS_PER_SPLIT:
            rows = rows.sample(MAX_ROWS_PER_SPLIT, random_state=0)
            notes.append(
                f"The {split} split was subsampled to {MAX_ROWS_PER_SPLIT:,} rows for "
                "this check, so the counts below are a lower bound."
            )
        frames[split] = rows

    train_df, holdout_df = frames["train"], frames[holdout_split]
    if train_df.empty or holdout_df.empty:
        return skipped(key, title, "One of the splits is empty.", "")

    # Exact duplicates, computed on the full frame rather than the subsample.
    prints = fingerprint(df, dataset.feature_cols)
    split_series = df[dataset.split_col]
    spanning = (
        pd.DataFrame({"fp": prints, "split": split_series})
        .groupby("fp")["split"]
        .nunique()
    )
    exact_fps = set(spanning[spanning > 1].index)
    exact_rows = df[prints.isin(exact_fps)] if exact_fps else df.iloc[0:0]
    exact_holdout = exact_rows[exact_rows[dataset.split_col] == holdout_split]

    # Near duplicates via cosine nearest neighbour.
    combined = pd.concat([train_df, holdout_df])
    features = build_matrix(combined, dataset.feature_cols)
    if features.is_empty:
        return skipped(
            key,
            title,
            "Feature columns could not be encoded into comparable values.",
            "Check that feature columns hold numbers, categories or text rather than "
            "entirely empty values.",
        )

    if features.matrix.shape[1] < MIN_DIMS_FOR_SIMILARITY:
        return _low_dimensional_result(
            key, title, features.matrix.shape[1], len(exact_holdout), len(holdout_df)
        )

    train_matrix = features.matrix[: len(train_df)]
    holdout_matrix = features.matrix[len(train_df) :]

    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(train_matrix)
    distances, indices = nn.kneighbors(holdout_matrix)
    sims = 1.0 - distances.ravel()
    neighbour_idx = indices.ravel()

    # Calibrate against ordinary pairs so the cut-off adapts to this feature space.
    background = _background_similarity(features.matrix)
    effective = threshold
    if background is not None and background > threshold:
        effective = background
        notes.append(
            f"In this dataset even unrelated rows score up to {background:.3f} cosine "
            f"similarity, so the threshold was raised from {threshold:.2f} to {effective:.3f}. "
            "Only pairs that are genuine outliers against the dataset's own distribution "
            "are counted."
        )

    flagged_mask = sims >= effective
    flagged_count = int(flagged_mask.sum())
    fraction = flagged_count / len(holdout_df)

    metrics = {
        "Held-out rows checked": f"{len(holdout_df):,}",
        "Near-duplicates found": f"{flagged_count:,}",
        "Share of held-out set": f"{fraction:.1%}",
        "Exact duplicates across splits": f"{len(exact_holdout):,}",
        "Similarity threshold used": f"{effective:.3f}",
    }
    if background is not None:
        metrics["Typical unrelated pair"] = f"{background:.3f}"

    # What the score looks like once leaked rows are dropped from the held-out set.
    full_acc = _accuracy(holdout_df, dataset)
    clean_acc = _accuracy(holdout_df[~flagged_mask], dataset)
    accuracy_line = ""
    if full_acc is not None and clean_acc is not None and flagged_count:
        drop = (full_acc - clean_acc) * 100
        metrics["Accuracy on full held-out set"] = f"{full_acc:.1%}"
        metrics["Accuracy excluding leaked rows"] = f"{clean_acc:.1%}"
        metrics["Difference"] = f"{drop:+.1f} pts"
        if drop >= MIN_REPORTABLE_DROP_PTS:
            accuracy_line = (
                f" Excluding them, accuracy moves from {full_acc:.1%} to {clean_acc:.1%} "
                f"({drop:+.1f} points)."
            )
        else:
            # Deleting duplicate rows barely moved the score. That is not reassurance:
            # it usually means the contamination is at a level above the row.
            notes.append(
                f"Dropping the flagged rows changes accuracy by {drop:+.1f} points "
                f"({full_acc:.1%} to {clean_acc:.1%}), which is not the improvement you might "
                "expect. Deleting rows is not the fix here: the rows that remain are still "
                "drawn from the same underlying entities as the training set, so the whole "
                "held-out set is affected rather than just the flagged rows. Re-splitting so "
                "that entire groups stay on one side is the fix - check the split-integrity "
                "result below for which grouping is being violated."
            )

    tables = []
    pair_rows = _pair_rows(train_df, holdout_df, sims, neighbour_idx, effective, dataset)
    if pair_rows:
        tables.append(
            {
                "title": f"Most similar pairs (showing {len(pair_rows)} of {flagged_count:,})",
                "columns": ["similarity", "holdout_id", "train_id", "holdout_label", "train_label"],
                "rows": pair_rows,
            }
        )

    if flagged_count == 0:
        return CheckResult(
            key=key,
            title=title,
            status=Status.PASS,
            headline=(
                f"No held-out sample is more than {effective:.1%} similar to a training "
                f"sample. The closest pair scores {sims.max():.3f}."
            ),
            why_it_matters=(
                "A clean separation means the held-out score is measuring generalisation "
                "rather than recall."
            ),
            metrics=metrics,
            notes=notes,
        )

    status = Status.FAIL if fraction >= FAIL_FRACTION or len(exact_holdout) else Status.WARN
    return CheckResult(
        key=key,
        title=title,
        status=status,
        headline=(
            f"{flagged_count:,} of {len(holdout_df):,} held-out samples ({fraction:.1%}) have a "
            f"near-identical twin in the training set." + accuracy_line
        ),
        why_it_matters=(
            "The model saw these samples, or something indistinguishable from them, during "
            "training. Getting them right demonstrates memory, not generalisation, so every "
            "leaked row inflates the reported score. This is easy to create by accident: "
            "splitting video frames, augmented copies, or repeated records at the row level "
            "puts the same underlying thing on both sides of the split."
        ),
        what_to_do=(
            "Find what these pairs have in common - usually a source document, subject, "
            "session or original image that was split across train and test. Add that "
            "identifier as a group column and re-split so an entire group lands on one side. "
            "Then re-run training and quote the new number."
        ),
        metrics=metrics,
        tables=tables,
        notes=notes,
    )
