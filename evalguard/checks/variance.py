"""Check 3 - variance across seeds and across held-out sets.

The failure this catches: quoting a single number from a single run as though it
were a property of the method. Re-running with a different seed, or holding out a
different slice of the data, moves the number - sometimes by more than the
improvement being claimed.

The distinction this check draws is the important one. Repeating a run with a new
training seed while keeping the split fixed measures optimisation noise, and it is
usually small. Re-drawing the split so that genuinely different examples are held
out measures whether the result depends on which examples you happened to test on,
and it is usually much larger. A tight seed spread is often mistaken for evidence
of robustness; it is not, and comparing the two spreads side by side is the fastest
way to show that.
"""

from __future__ import annotations

import math
import re
import statistics

from evalguard.checks.base import CheckResult, Status, skipped

# Spread of the held-out scores, in percentage points, that escalates the verdict.
WARN_RANGE_PTS = 2.0
FAIL_RANGE_PTS = 5.0
# How many times larger the split spread must be before it is worth calling out.
RATIO_CALLOUT = 3.0


def parse_scores(text: str) -> list[float]:
    """Pull numbers out of whatever the user pasted.

    Accepts commas, spaces, newlines, and lines like `seed 1000: acc 0.8171`.
    Values above 1 are read as percentages so 81.71 and 0.8171 both work.
    """
    if not text:
        return []
    found = re.findall(r"-?\d+\.?\d*", text)
    scores = []
    for raw in found:
        try:
            value = float(raw)
        except ValueError:
            continue
        # Integers like a seed number (1000) or an epoch are not scores.
        if value > 100:
            continue
        scores.append(value / 100 if value > 1 else value)
    return scores


def _stats(scores: list[float]) -> dict:
    mean = statistics.fmean(scores)
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    return {
        "n": len(scores),
        "mean": mean,
        "sd": sd,
        "min": min(scores),
        "max": max(scores),
        "range": max(scores) - min(scores),
        # Standard error of the mean, the honest error bar on an average of n runs.
        "sem": sd / math.sqrt(len(scores)) if len(scores) > 1 else 0.0,
    }


def _describe(name: str, s: dict) -> dict:
    return {
        "group": name,
        "runs": s["n"],
        "mean": f"{s['mean']:.4f}",
        "std_dev": f"{s['sd']:.4f}",
        "range": f"{s['min']:.4f} to {s['max']:.4f}",
        "spread_pts": f"{s['range'] * 100:.2f}",
    }


def run(seed_scores: list[float], split_scores: list[float]) -> CheckResult:
    key, title = "variance", "Run-to-run variance"

    if not seed_scores and not split_scores:
        return skipped(
            key,
            title,
            "No repeated-run scores supplied, so variance could not be measured.",
            "Re-run your evaluation with at least three different training seeds, and "
            "again with three different train/test splits, then paste both sets of "
            "scores in. A single run cannot tell you how noisy it is.",
        )

    tables, metrics, rows = [], {}, []
    seed_stats = _stats(seed_scores) if seed_scores else None
    split_stats = _stats(split_scores) if split_scores else None

    if seed_stats:
        rows.append(_describe("Different training seeds, split fixed", seed_stats))
        metrics["Seed runs"] = str(seed_stats["n"])
        metrics["Seed spread"] = f"{seed_stats['range'] * 100:.2f} pts"
        metrics["Seed std dev"] = f"{seed_stats['sd'] * 100:.2f} pts"
    if split_stats:
        rows.append(_describe("Different held-out sets", split_stats))
        metrics["Split runs"] = str(split_stats["n"])
        metrics["Split spread"] = f"{split_stats['range'] * 100:.2f} pts"
        metrics["Split std dev"] = f"{split_stats['sd'] * 100:.2f} pts"
        metrics["Mean across splits"] = f"{split_stats['mean']:.1%}"

    tables.append(
        {
            "title": "Spread by what was varied",
            "columns": ["group", "runs", "mean", "std_dev", "range", "spread_pts"],
            "rows": rows,
        }
    )

    # The headline comparison: optimisation noise versus data-dependence.
    if seed_stats and split_stats and seed_stats["sd"] > 0:
        ratio = split_stats["sd"] / seed_stats["sd"]
        metrics["Split noise / seed noise"] = f"{ratio:.0f}x"

        if ratio >= RATIO_CALLOUT:
            return CheckResult(
                key=key,
                title=title,
                status=Status.FAIL,
                headline=(
                    f"Changing the training seed moves the score by "
                    f"{seed_stats['sd'] * 100:.2f} points, but changing which examples are "
                    f"held out moves it by {split_stats['sd'] * 100:.2f} points - "
                    f"{ratio:.0f} times more."
                ),
                why_it_matters=(
                    "Low seed variance is frequently reported as evidence that a result is "
                    "stable, and it is not. It only shows the optimiser lands in the same "
                    "place given the same data. The much larger spread across held-out sets "
                    "shows the score is a property of which examples happened to be in the "
                    f"test set, not of the method. Quoting any single run here implies a "
                    f"precision of a fraction of a point when the real uncertainty spans "
                    f"{split_stats['range'] * 100:.1f} points "
                    f"({split_stats['min']:.1%} to {split_stats['max']:.1%})."
                ),
                what_to_do=(
                    f"Report the mean across held-out sets with its spread - "
                    f"{split_stats['mean']:.1%} plus or minus {split_stats['sd'] * 100:.1f} points "
                    f"over {split_stats['n']} splits - rather than a single number. Then find "
                    "out which held-out examples drive the low end; a single hard class or "
                    "group is often responsible, and that is a finding worth reporting in "
                    "its own right."
                ),
                metrics=metrics,
                tables=tables,
            )

    primary = split_stats or seed_stats
    label = "held-out sets" if split_stats else "training seeds"
    spread_pts = primary["range"] * 100

    if primary["n"] < 3:
        status = Status.WARN
        headline = (
            f"Only {primary['n']} run(s) supplied, which is too few to estimate variance."
        )
        why = (
            "Two runs can differ by chance in either direction. Three or more give a spread "
            "you can actually quote."
        )
        what = "Re-run with at least three seeds and three different splits."
    elif spread_pts >= FAIL_RANGE_PTS:
        status = Status.FAIL
        headline = (
            f"Scores across {primary['n']} {label} span {spread_pts:.1f} points "
            f"({primary['min']:.1%} to {primary['max']:.1%})."
        )
        why = (
            "A spread this wide means a single run is not a reliable estimate. Any improvement "
            f"smaller than {spread_pts:.1f} points could be produced by re-running alone, so "
            "comparisons against a baseline at that scale are not meaningful."
        )
        what = (
            f"Report the mean and spread ({primary['mean']:.1%} plus or minus "
            f"{primary['sd'] * 100:.1f} points) instead of a single number, and treat "
            "differences smaller than that as noise."
        )
    elif spread_pts >= WARN_RANGE_PTS:
        status = Status.WARN
        headline = (
            f"Scores across {primary['n']} {label} span {spread_pts:.1f} points "
            f"({primary['min']:.1%} to {primary['max']:.1%})."
        )
        why = (
            "That is a moderate spread. It is small enough to quote a mean, but large enough "
            "that improvements of a point or two are not distinguishable from noise."
        )
        what = (
            f"Quote {primary['mean']:.1%} plus or minus {primary['sd'] * 100:.1f} points, and "
            "make sure any claimed gain is larger than that."
        )
    else:
        status = Status.PASS
        headline = (
            f"Scores across {primary['n']} {label} span only {spread_pts:.1f} points "
            f"({primary['min']:.1%} to {primary['max']:.1%})."
        )
        why = "The result is reproducible at this level of variation."
        what = (
            f"Quote {primary['mean']:.1%} plus or minus {primary['sd'] * 100:.1f} points."
            + (
                " You have only varied the training seed - vary the split too, since that is "
                "usually the larger source of variance."
                if not split_stats
                else ""
            )
        )

    if not split_stats:
        status = Status.WARN if status == Status.PASS else status

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
