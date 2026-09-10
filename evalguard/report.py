"""Run every check and assemble the report."""

from __future__ import annotations

from evalguard.checks import Status
from evalguard.checks import baseline, integrity, leakage, variance
from evalguard.loader import Dataset

# Checks run in the order they are listed in the README.
VERDICT_TEXT = {
    Status.FAIL: (
        "This evaluation has at least one fault serious enough that the reported score "
        "should not be quoted until it is fixed."
    ),
    Status.WARN: (
        "This evaluation is broadly sound, but something below will distort the number "
        "or overstate its precision."
    ),
    Status.PASS: (
        "No evaluation faults detected. The reported score is measuring what it claims to."
    ),
    Status.SKIP: "Not enough information to reach a verdict - see the skipped checks below.",
}


def analyse(
    dataset: Dataset,
    seed_scores: list[float] | None = None,
    split_scores: list[float] | None = None,
    similarity_threshold: float = leakage.DEFAULT_NEAR_THRESHOLD,
) -> dict:
    results = [
        leakage.run(dataset, threshold=similarity_threshold),
        baseline.run(dataset),
        variance.run(seed_scores or [], split_scores or []),
        integrity.run(dataset),
    ]

    counts = {s: 0 for s in (Status.PASS, Status.WARN, Status.FAIL, Status.SKIP)}
    for r in results:
        counts[r.status] += 1

    if counts[Status.FAIL]:
        verdict = Status.FAIL
    elif counts[Status.WARN]:
        verdict = Status.WARN
    elif counts[Status.PASS]:
        verdict = Status.PASS
    else:
        verdict = Status.SKIP

    return {
        "verdict": verdict,
        "verdict_text": VERDICT_TEXT[verdict],
        "counts": counts,
        "dataset": dataset.summary(),
        "warnings": dataset.warnings,
        "checks": [r.to_dict() for r in results],
    }
