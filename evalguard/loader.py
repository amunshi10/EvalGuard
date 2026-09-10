"""CSV ingest and schema detection.

The input contract is deliberately small (see README): a CSV where one row is
one evaluation sample. Only `split` and `label` are strictly required. Column
names are matched case-insensitively against a list of common aliases so people
do not have to rename anything before their first run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Aliases are checked in order; first hit wins.
ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "sample_id", "example_id", "uid", "index", "idx", "row_id", "image_id", "filename", "file"),
    "split": ("split", "partition", "fold", "subset", "set", "dataset_split"),
    "label": ("label", "target", "y", "class", "ground_truth", "gt", "y_true", "actual", "true_label"),
    "prediction": ("prediction", "pred", "y_pred", "predicted", "predicted_label", "model_prediction", "output"),
    "group": ("group", "doc_id", "document_id", "video_id", "subject_id", "patient_id", "user_id", "source_id", "session_id", "identity"),
}

# Raw split values are normalised so "Training"/"TRAIN"/"tr" all collapse together.
SPLIT_NORMALISATION: dict[str, str] = {
    "train": "train", "training": "train", "tr": "train", "fit": "train", "0": "train",
    "test": "test", "testing": "test", "te": "test", "eval": "test", "evaluation": "test", "holdout": "test", "hold-out": "test", "2": "test",
    "val": "val", "valid": "val", "validation": "val", "dev": "val", "1": "val",
}


class DatasetError(ValueError):
    """Raised when the CSV cannot be interpreted at all."""


@dataclass
class Dataset:
    df: pd.DataFrame
    id_col: str | None
    split_col: str
    label_col: str
    pred_col: str | None
    group_col: str | None
    feature_cols: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def splits(self) -> list[str]:
        """Split names present, ordered train -> val -> test -> anything else."""
        preferred = ["train", "val", "test"]
        present = list(self.df[self.split_col].dropna().unique())
        ordered = [s for s in preferred if s in present]
        ordered += sorted(s for s in present if s not in preferred)
        return ordered

    def rows_for(self, split: str) -> pd.DataFrame:
        return self.df[self.df[self.split_col] == split]

    @property
    def holdout_split(self) -> str | None:
        """The split whose score people actually quote in a paper or README."""
        for candidate in ("test", "val"):
            if candidate in self.splits:
                return candidate
        non_train = [s for s in self.splits if s != "train"]
        return non_train[0] if non_train else None

    def summary(self) -> dict:
        counts = self.df[self.split_col].value_counts().to_dict()
        return {
            "rows": int(len(self.df)),
            "splits": {str(k): int(v) for k, v in counts.items()},
            "classes": int(self.df[self.label_col].nunique()),
            "feature_columns": len(self.feature_cols),
            "has_predictions": self.pred_col is not None,
            "has_groups": self.group_col is not None,
            "id_column": self.id_col,
            "label_column": self.label_col,
            "split_column": self.split_col,
            "prediction_column": self.pred_col,
            "group_column": self.group_col,
        }


def _find_column(columns: list[str], role: str) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for alias in ALIASES[role]:
        if alias in lowered:
            return lowered[alias]
    return None


def normalise_split_value(value) -> str:
    if pd.isna(value):
        return "unlabelled"
    text = str(value).strip().lower()
    return SPLIT_NORMALISATION.get(text, text)


def _resolve(columns: list[str], name: str | None) -> str | None:
    """Match a user-supplied column name case-insensitively."""
    if not name:
        return None
    lowered = {c.lower().strip(): c for c in columns}
    return lowered.get(name.lower().strip())


def load_dataset(
    source,
    max_rows: int = 100_000,
    group_override: str | None = None,
    ignore_cols: list[str] | None = None,
) -> Dataset:
    """Read a CSV (path or file-like) into a validated Dataset.

    `group_override` names the column holding the real-world entity a row belongs
    to - a document, patient or session. Auto-detection only recognises common
    names, and choosing the right grouping level is the most consequential
    decision in setting up a split, so it is worth stating explicitly.

    `ignore_cols` are columns excluded from the feature set: metadata that
    describes a row without being an input the model actually saw.
    """
    try:
        df = pd.read_csv(source)
    except Exception as exc:  # pragma: no cover - surfaced straight to the user
        raise DatasetError(f"Could not parse that file as CSV: {exc}") from exc

    if df.empty:
        raise DatasetError("The file parsed correctly but contains no rows.")

    warnings: list[str] = []
    if len(df) > max_rows:
        warnings.append(
            f"File has {len(df):,} rows; EvalGuard analysed the first {max_rows:,}."
        )
        df = df.head(max_rows).copy()

    columns = list(df.columns)
    split_col = _find_column(columns, "split")
    label_col = _find_column(columns, "label")

    if split_col is None:
        raise DatasetError(
            "No split column found. Add a column named 'split' whose values are "
            "train / test (val is optional). Aliases accepted: "
            + ", ".join(ALIASES["split"])
        )
    if label_col is None:
        raise DatasetError(
            "No label column found. Add a column named 'label' holding the ground "
            "truth class. Aliases accepted: " + ", ".join(ALIASES["label"])
        )

    id_col = _find_column(columns, "id")
    pred_col = _find_column(columns, "prediction")

    group_col = _find_column(columns, "group")
    if group_override:
        resolved = _resolve(columns, group_override)
        if resolved is None:
            warnings.append(
                f"Requested group column '{group_override}' is not in this file; "
                "fell back to auto-detection."
            )
        else:
            group_col = resolved

    df = df.copy()
    df[split_col] = df[split_col].map(normalise_split_value)

    ignored: list[str] = []
    for name in ignore_cols or []:
        resolved = _resolve(columns, name)
        if resolved:
            ignored.append(resolved)

    reserved = {c for c in (id_col, split_col, label_col, pred_col, group_col) if c}
    reserved.update(ignored)
    feature_cols = [c for c in columns if c not in reserved]

    if ignored:
        warnings.append(
            "Excluded from the feature comparison: " + ", ".join(sorted(ignored)) + "."
        )
    if group_col is None:
        warnings.append(
            "No group column identified. If your rows are not independent - repeated "
            "measurements, video frames, multiple records per person - name the column "
            "that identifies the underlying entity so EvalGuard can check whether one "
            "straddles the split."
        )

    if id_col is None:
        warnings.append(
            "No id column found - row numbers are used instead, so the duplicate-ID "
            "check is limited to feature values."
        )
    if pred_col is None:
        warnings.append(
            "No prediction column found - EvalGuard can report the naive baseline but "
            "cannot compare your model against it. Add a 'prediction' column to unlock that."
        )
    if not feature_cols:
        warnings.append(
            "No feature columns found - the leakage check needs at least one feature "
            "column to compare samples against each other."
        )

    unknown = sorted(
        s for s in df[split_col].unique() if s not in {"train", "val", "test", "unlabelled"}
    )
    if unknown:
        warnings.append(
            "Unrecognised split values treated as their own splits: " + ", ".join(map(str, unknown))
        )

    return Dataset(
        df=df,
        id_col=id_col,
        split_col=split_col,
        label_col=label_col,
        pred_col=pred_col,
        group_col=group_col,
        feature_cols=feature_cols,
        warnings=warnings,
    )
