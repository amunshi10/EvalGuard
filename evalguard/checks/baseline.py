"""Check 2 - class imbalance versus the naive baseline.

The failure this catches: a headline accuracy that sounds strong but is close to
what you would get by ignoring the input entirely and always guessing the most
common class. On a 92/8 split, 92% accuracy is not a result - it is the dataset
description.
"""

from __future__ import annotations

import math

import pandas as pd

from evalguard.checks.base import CheckResult, Status, skipped
from evalguard.loader import Dataset

# Gap between model accuracy and majority-class accuracy, in percentage points.
FAIL_MARGIN_PTS = 2.0
WARN_MARGIN_PTS = 10.0
# A dataset this skewed makes plain accuracy a misleading headline metric.
IMBALANCE_WARN = 0.80
# A model predicting one class this often has effectively collapsed.
DEGENERATE_SHARE = 0.95


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval - honest error bars for an accuracy on n samples."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _per_class_table(df: pd.DataFrame, dataset: Dataset) -> dict:
    truth = df[dataset.label_col].astype(str)
    rows = []
    if dataset.pred_col is not None:
        pred = df[dataset.pred_col].astype(str)
        for cls, count in truth.value_counts().items():
            mask = truth == cls
            recall = float((pred[mask] == cls).mean()) if mask.any() else 0.0
            predicted_as = int((pred == cls).sum())
            precision = float((truth[pred == cls] == cls).mean()) if predicted_as else 0.0
            rows.append(
                {
                    "class": cls,
                    "held_out_count": int(count),
                    "share": f"{count / len(df):.1%}",
                    "recall": f"{recall:.1%}",
                    "precision": f"{precision:.1%}" if predicted_as else "n/a",
                    "times_predicted": predicted_as,
                }
            )
        columns = ["class", "held_out_count", "share", "recall", "precision", "times_predicted"]
    else:
        for cls, count in truth.value_counts().items():
            rows.append(
                {"class": cls, "held_out_count": int(count), "share": f"{count / len(df):.1%}"}
            )
        columns = ["class", "held_out_count", "share"]

    return {"title": "Per-class breakdown of the held-out set", "columns": columns, "rows": rows}


def _macro_f1(df: pd.DataFrame, dataset: Dataset) -> float:
    truth = df[dataset.label_col].astype(str)
    pred = df[dataset.pred_col].astype(str)
    scores = []
    for cls in truth.unique():
        tp = int(((pred == cls) & (truth == cls)).sum())
        fp = int(((pred == cls) & (truth != cls)).sum())
        fn = int(((pred != cls) & (truth == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def run(dataset: Dataset) -> CheckResult:
    key, title = "baseline", "Accuracy versus naive baseline"

    holdout_split = dataset.holdout_split
    if holdout_split is None:
        return skipped(key, title, "No held-out split to evaluate.", "")

    df = dataset.rows_for(holdout_split)
    if df.empty:
        return skipped(key, title, "The held-out split is empty.", "")

    truth = df[dataset.label_col].astype(str)
    counts = truth.value_counts()
    majority_class = str(counts.index[0])
    baseline = float(counts.iloc[0] / len(df))

    metrics = {
        "Held-out rows": f"{len(df):,}",
        "Classes": f"{truth.nunique()}",
        "Majority class": majority_class,
        "Always-guess-majority accuracy": f"{baseline:.1%}",
    }
    tables = [_per_class_table(df, dataset)]

    # Without predictions we can still tell them what bar they have to clear.
    if dataset.pred_col is None:
        status = Status.WARN if baseline >= IMBALANCE_WARN else Status.PASS
        headline = (
            f"Always guessing '{majority_class}' would score {baseline:.1%} on this held-out set."
        )
        return CheckResult(
            key=key,
            title=title,
            status=status,
            headline=headline,
            why_it_matters=(
                "This is the bar any reported accuracy has to clear before it means anything. "
                + (
                    "At this level of imbalance, accuracy is a poor headline metric - a model "
                    "can look strong while never once getting the minority class right, which "
                    "is usually the class you actually care about."
                    if baseline >= IMBALANCE_WARN
                    else "The classes are reasonably balanced, so accuracy is a fair summary here."
                )
            ),
            what_to_do=(
                "Add a 'prediction' column so EvalGuard can compare your model against this "
                "baseline directly"
                + (
                    ", and report macro-F1 or per-class recall alongside accuracy."
                    if baseline >= IMBALANCE_WARN
                    else "."
                )
            ),
            metrics=metrics,
            tables=tables,
        )

    pred = df[dataset.pred_col].astype(str)
    correct = int((pred == truth).sum())
    accuracy = correct / len(df)
    margin_pts = (accuracy - baseline) * 100
    lo, hi = wilson_interval(correct, len(df))
    macro_f1 = _macro_f1(df, dataset)
    top_pred_share = float(pred.value_counts().iloc[0] / len(pred))

    metrics.update(
        {
            "Model accuracy": f"{accuracy:.1%}",
            "95% confidence interval": f"{lo:.1%} to {hi:.1%}",
            "Gain over baseline": f"{margin_pts:+.1f} pts",
            "Macro-F1": f"{macro_f1:.3f}",
        }
    )

    degenerate = top_pred_share >= DEGENERATE_SHARE and truth.nunique() > 1
    if degenerate:
        metrics["Most-predicted class share"] = f"{top_pred_share:.1%}"

    if margin_pts < FAIL_MARGIN_PTS or degenerate:
        status = Status.FAIL
    elif margin_pts < WARN_MARGIN_PTS or baseline >= IMBALANCE_WARN:
        status = Status.WARN
    else:
        status = Status.PASS

    if degenerate:
        headline = (
            f"The model predicts '{str(pred.value_counts().index[0])}' for {top_pred_share:.1%} of "
            f"the held-out set, scoring {accuracy:.1%} against a {baseline:.1%} baseline."
        )
        why = (
            "A model that almost always emits one class has collapsed to the majority class. "
            "Its accuracy is tracking the class distribution, not the input. Macro-F1 of "
            f"{macro_f1:.3f} confirms the minority classes are being handled poorly."
        )
        what = (
            "Check for a training problem before trusting anything downstream: class weighting, "
            "learning rate, or a label that is not actually predictable from these features. "
            "Report macro-F1 and per-class recall, not accuracy."
        )
    elif margin_pts < FAIL_MARGIN_PTS:
        headline = (
            f"The model scores {accuracy:.1%}, only {margin_pts:+.1f} points above the "
            f"{baseline:.1%} you would get by always guessing '{majority_class}'."
        )
        why = (
            "That gap is the entire value the model adds over a constant guess. At this size it "
            "is within, or close to, the noise of a test set this large - the confidence interval "
            f"alone spans {(hi - lo) * 100:.1f} points. The headline accuracy is mostly describing "
            "how imbalanced the data is."
        )
        what = (
            "Report the gain over baseline rather than raw accuracy, and switch the headline "
            "metric to macro-F1 or per-class recall. If the gain does not grow, the model is not "
            "learning a useful signal from these features."
        )
    else:
        headline = (
            f"The model scores {accuracy:.1%}, {margin_pts:+.1f} points above the {baseline:.1%} "
            f"majority-class baseline."
        )
        why = (
            "The model is clearly doing better than a constant guess"
            + (
                ". Note the class imbalance though: with this distribution, accuracy still hides "
                "how the minority classes are doing, so quote macro-F1 alongside it."
                if baseline >= IMBALANCE_WARN
                else ", and the classes are balanced enough that accuracy is a fair summary."
            )
        )
        what = (
            "Quote the baseline next to your accuracy so a reader can see the size of the gain, "
            f"and quote the confidence interval ({lo:.1%} to {hi:.1%}) so they can see its precision."
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
