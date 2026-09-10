"""Check 4 - split integrity.

The failure this catches: the split itself is malformed in a way that makes the
score unreadable, independently of the model. Ids repeated across train and test,
a group (document, patient, session) straddling the boundary, a class that appears
in test but was never trained on, or a test set so small that its error bars swamp
any result.

The group-overlap test is the one that usually fires. Row-level splitting is the
default in every tutorial, and it silently breaks whenever rows are not independent
- video frames, augmented copies, repeated visits, multiple records per customer.
"""

from __future__ import annotations

import pandas as pd

from evalguard.checks.base import CheckResult, Status
from evalguard.loader import Dataset

# Held-out sets below this size have error bars too wide to compare models with.
SMALL_TEST_WARN = 200
SMALL_TEST_FAIL = 50
# Total variation distance between train and test label distributions.
DRIFT_WARN = 0.10
DRIFT_FAIL = 0.25
MAX_ROWS_SHOWN = 20


def _label_drift(train: pd.Series, holdout: pd.Series) -> tuple[float, list[dict]]:
    """Total variation distance plus a per-class comparison table."""
    train_dist = train.value_counts(normalize=True)
    holdout_dist = holdout.value_counts(normalize=True)
    classes = sorted(set(train_dist.index) | set(holdout_dist.index))

    rows = []
    tvd = 0.0
    for cls in classes:
        p = float(train_dist.get(cls, 0.0))
        q = float(holdout_dist.get(cls, 0.0))
        tvd += abs(p - q)
        rows.append(
            {
                "class": str(cls),
                "train_share": f"{p:.1%}",
                "held_out_share": f"{q:.1%}",
                "difference": f"{(q - p) * 100:+.1f} pts",
            }
        )
    return tvd / 2, rows


def run(dataset: Dataset) -> CheckResult:
    key, title = "integrity", "Split integrity"

    df = dataset.df
    problems: list[str] = []
    tables: list[dict] = []
    metrics: dict[str, str] = {}
    status = Status.PASS

    def escalate(level: str) -> None:
        nonlocal status
        if level == Status.FAIL or status == Status.FAIL:
            status = Status.FAIL
        elif level == Status.WARN:
            status = Status.WARN

    holdout_split = dataset.holdout_split
    train_df = dataset.rows_for("train") if "train" in dataset.splits else df.iloc[0:0]
    holdout_df = dataset.rows_for(holdout_split) if holdout_split else df.iloc[0:0]

    metrics["Rows"] = f"{len(df):,}"
    for split in dataset.splits:
        metrics[f"Rows in {split}"] = f"{len(dataset.rows_for(split)):,}"

    # 1. Duplicate ids, within a split and across splits.
    if dataset.id_col:
        ids = df[dataset.id_col].astype(str)
        spanning = df.assign(_id=ids).groupby("_id")[dataset.split_col].nunique()
        cross = set(spanning[spanning > 1].index)
        dup_total = int(ids.duplicated().sum())

        metrics["Duplicate ids"] = f"{dup_total:,}"
        metrics["Ids in more than one split"] = f"{len(cross):,}"

        if cross:
            escalate(Status.FAIL)
            problems.append(
                f"{len(cross):,} ids appear in more than one split - the same sample is "
                "being trained on and tested on."
            )
            sample = df[ids.isin(cross)].head(MAX_ROWS_SHOWN)
            tables.append(
                {
                    "title": "Ids appearing in more than one split",
                    "columns": ["id", "split", "label"],
                    "rows": [
                        {
                            "id": str(r[dataset.id_col]),
                            "split": str(r[dataset.split_col]),
                            "label": str(r[dataset.label_col]),
                        }
                        for _, r in sample.iterrows()
                    ],
                }
            )
        elif dup_total:
            escalate(Status.WARN)
            problems.append(
                f"{dup_total:,} duplicate ids exist within splits - ids are not unique, so "
                "every count here may be off."
            )

    # 2. Group overlap - the failure that row-level splitting creates.
    if dataset.group_col and holdout_split:
        train_groups = set(train_df[dataset.group_col].astype(str))
        holdout_groups = set(holdout_df[dataset.group_col].astype(str))
        overlap = train_groups & holdout_groups
        metrics[f"Distinct {dataset.group_col} values"] = f"{df[dataset.group_col].nunique():,}"
        metrics["Groups in both train and held-out"] = f"{len(overlap):,}"

        if overlap:
            escalate(Status.FAIL)
            share = len(overlap) / max(len(holdout_groups), 1)
            problems.append(
                f"{len(overlap):,} of {len(holdout_groups):,} held-out "
                f"'{dataset.group_col}' values ({share:.0%}) also appear in training."
            )
            counts = (
                holdout_df[holdout_df[dataset.group_col].astype(str).isin(overlap)][
                    dataset.group_col
                ]
                .astype(str)
                .value_counts()
                .head(MAX_ROWS_SHOWN)
            )
            tables.append(
                {
                    "title": f"'{dataset.group_col}' values on both sides of the split",
                    "columns": [dataset.group_col, "held_out_rows", "train_rows"],
                    "rows": [
                        {
                            dataset.group_col: str(g),
                            "held_out_rows": int(n),
                            "train_rows": int(
                                (train_df[dataset.group_col].astype(str) == g).sum()
                            ),
                        }
                        for g, n in counts.items()
                    ],
                }
            )

    # 3. Classes that cannot be learned or cannot be measured.
    if holdout_split and not train_df.empty and not holdout_df.empty:
        train_labels = train_df[dataset.label_col].astype(str)
        holdout_labels = holdout_df[dataset.label_col].astype(str)
        unseen = sorted(set(holdout_labels) - set(train_labels))
        untested = sorted(set(train_labels) - set(holdout_labels))

        if unseen:
            escalate(Status.FAIL)
            problems.append(
                f"{len(unseen)} class(es) appear in the held-out set but never in training: "
                + ", ".join(unseen[:5])
                + ("..." if len(unseen) > 5 else "")
                + ". The model cannot get these right."
            )
        if untested:
            escalate(Status.WARN)
            problems.append(
                f"{len(untested)} class(es) are trained on but never tested: "
                + ", ".join(untested[:5])
                + ("..." if len(untested) > 5 else "")
                + "."
            )

        drift, drift_rows = _label_drift(train_labels, holdout_labels)
        metrics["Label distribution shift"] = f"{drift:.1%}"
        if drift >= DRIFT_FAIL:
            escalate(Status.FAIL)
            problems.append(
                f"Train and held-out class distributions differ by {drift:.1%} - the split "
                "is not stratified and the two sets are not measuring the same task."
            )
        elif drift >= DRIFT_WARN:
            escalate(Status.WARN)
            problems.append(
                f"Train and held-out class distributions differ by {drift:.1%}."
            )
        tables.append(
            {
                "title": "Class balance, train versus held-out",
                "columns": ["class", "train_share", "held_out_share", "difference"],
                "rows": drift_rows,
            }
        )

    # 4. Is the held-out set big enough to say anything.
    if holdout_split:
        n = len(holdout_df)
        # Half-width of a 95% interval at the worst case p = 0.5.
        margin = 0.98 / (n**0.5) if n else 0.0
        metrics["Widest 95% error bar at this size"] = f"+/-{margin * 100:.1f} pts"
        if n < SMALL_TEST_FAIL:
            escalate(Status.FAIL)
            problems.append(
                f"The held-out set has only {n} rows, so any accuracy from it carries an "
                f"error bar of roughly +/-{margin * 100:.0f} points."
            )
        elif n < SMALL_TEST_WARN:
            escalate(Status.WARN)
            problems.append(
                f"The held-out set has {n} rows - small enough that differences under "
                f"{margin * 100:.0f} points are not measurable."
            )

    # 5. Rows that never got assigned to a split.
    unlabelled = int((df[dataset.split_col] == "unlabelled").sum())
    if unlabelled:
        escalate(Status.WARN)
        metrics["Rows with no split"] = f"{unlabelled:,}"
        problems.append(f"{unlabelled:,} rows have a missing split value and were excluded.")

    if status == Status.PASS:
        headline = (
            "The split is well formed: no shared ids"
            + (f" or {dataset.group_col} values" if dataset.group_col else "")
            + ", classes present on both sides, and a held-out set large enough to measure with."
        )
        why = (
            "These are the structural faults that make a score unreadable before the model is "
            "even involved. None of them are present."
        )
        what = "Nothing to fix here."
    else:
        headline = problems[0] if problems else "Problems found in the split."
        if len(problems) > 1:
            headline += f" ({len(problems) - 1} more issue(s) below.)"
        why = (
            "A split has to satisfy some basic structural properties before the number it "
            "produces means anything: each sample on exactly one side, each real-world entity "
            "on exactly one side, every class represented on both, and enough held-out rows "
            "for the result to be more than noise. "
            + (
                "Group overlap is the subtle one - splitting rows at random looks correct "
                "and quietly puts the same underlying document, person or session on both "
                "sides."
                if dataset.group_col
                else "Without a group column, EvalGuard cannot check whether the same "
                "real-world entity appears on both sides - add one if your rows are not "
                "independent."
            )
        )
        what = (
            "Fix these before reading any other check. "
            + (
                "Re-split so entire groups land on one side (scikit-learn's GroupShuffleSplit "
                "or GroupKFold do this), then re-train. "
                if dataset.group_col
                else ""
            )
            + "Stratify by label so both sides see the same class mix."
        )

    if len(problems) > 1:
        tables.insert(
            0,
            {
                "title": "All issues found",
                "columns": ["issue"],
                "rows": [{"issue": p} for p in problems],
            },
        )

    return CheckResult(
        key=key,
        title=title,
        status=status,
        headline=headline,
        why_it_matters=why,
        what_to_do=what,
        metrics=metrics,
        tables=tables,
    )
