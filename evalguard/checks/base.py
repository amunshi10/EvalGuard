"""Shared result type for every check.

Every check returns the same shape so the UI can render them uniformly and so
adding a new check never requires touching the front end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Status:
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"

    # Used to sort the results panel worst-first.
    ORDER = {FAIL: 0, WARN: 1, PASS: 2, SKIP: 3}


@dataclass
class CheckResult:
    """One check's verdict, written for a human rather than a log file.

    headline        - one sentence stating what was found, with the number in it.
    why_it_matters  - why this pattern inflates or distorts a reported score.
    what_to_do      - the concrete next action.
    metrics         - small key/value facts rendered as a stat strip.
    tables          - list of {"title", "columns", "rows"} rendered as detail tables.
    notes           - caveats about how the check itself ran (sampling, fallbacks).
    """

    key: str
    title: str
    status: str
    headline: str
    why_it_matters: str = ""
    what_to_do: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    tables: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "headline": self.headline,
            "why_it_matters": self.why_it_matters,
            "what_to_do": self.what_to_do,
            "metrics": self.metrics,
            "tables": self.tables,
            "notes": self.notes,
        }


def skipped(key: str, title: str, reason: str, what_to_do: str = "") -> CheckResult:
    """A check that could not run - never silently drop it from the report."""
    return CheckResult(
        key=key,
        title=title,
        status=Status.SKIP,
        headline=reason,
        what_to_do=what_to_do,
    )
