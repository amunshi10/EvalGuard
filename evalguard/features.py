"""Turn arbitrary feature columns into one matrix suitable for cosine similarity.

Real datasets mix numeric columns (embeddings, measurements), free text
(filenames, captions) and low-cardinality categoricals. Each block is encoded
separately, L2-normalised so no block dominates purely because it has more
columns, then stacked. The final rows are L2-normalised so a dot product is a
cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# A column with more distinct values than this is treated as free text, not a category.
MAX_CATEGORY_CARDINALITY = 50


@dataclass
class FeatureMatrix:
    matrix: sparse.csr_matrix
    numeric_cols: list[str]
    categorical_cols: list[str]
    text_cols: list[str]

    @property
    def description(self) -> str:
        parts = []
        if self.numeric_cols:
            parts.append(f"{len(self.numeric_cols)} numeric")
        if self.categorical_cols:
            parts.append(f"{len(self.categorical_cols)} categorical")
        if self.text_cols:
            parts.append(f"{len(self.text_cols)} text")
        return ", ".join(parts) if parts else "no usable features"

    @property
    def is_empty(self) -> bool:
        return self.matrix.shape[1] == 0


def classify_columns(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str], list[str]]:
    numeric, categorical, text = [], [], []
    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric.append(col)
            continue
        # A numeric column read as strings (thousands separators, stray spaces)
        # should still be treated as numeric rather than as text.
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().mean() > 0.9:
            numeric.append(col)
        elif series.nunique(dropna=True) <= MAX_CATEGORY_CARDINALITY:
            categorical.append(col)
        else:
            text.append(col)
    return numeric, categorical, text


def build_matrix(df: pd.DataFrame, feature_cols: list[str]) -> FeatureMatrix:
    numeric_cols, categorical_cols, text_cols = classify_columns(df, feature_cols)
    blocks: list[sparse.csr_matrix] = []

    if numeric_cols:
        num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        num = num.fillna(num.median(numeric_only=True)).fillna(0.0)
        values = num.to_numpy(dtype=float)
        # Standardise per column so a feature measured in thousands does not
        # drown out one measured in fractions.
        std = values.std(axis=0)
        std[std == 0] = 1.0
        values = (values - values.mean(axis=0)) / std
        blocks.append(normalize(sparse.csr_matrix(values)))

    if categorical_cols:
        dummies = pd.get_dummies(
            df[categorical_cols].astype(str), columns=categorical_cols, dtype=float
        )
        if dummies.shape[1]:
            blocks.append(normalize(sparse.csr_matrix(dummies.to_numpy(dtype=float))))

    if text_cols:
        joined = df[text_cols].astype(str).agg(" ".join, axis=1)
        # Character n-grams catch near-identical strings that differ only by a
        # frame number or a suffix - the shape of duplicated media filenames.
        vectoriser = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=20_000
        )
        try:
            blocks.append(normalize(vectoriser.fit_transform(joined)))
        except ValueError:
            pass  # Every value was empty or stop-word only; drop the block.

    if not blocks:
        return FeatureMatrix(
            sparse.csr_matrix((len(df), 0)), numeric_cols, categorical_cols, text_cols
        )

    stacked = normalize(sparse.hstack(blocks, format="csr"))
    return FeatureMatrix(stacked, numeric_cols, categorical_cols, text_cols)


def fingerprint(df: pd.DataFrame, feature_cols: list[str], precision: int = 6) -> pd.Series:
    """A stable string per row used for exact-duplicate detection.

    Floats are rounded so that values differing only by float noise still count
    as the same sample.
    """
    if not feature_cols:
        return pd.Series([""] * len(df), index=df.index)

    parts = []
    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_float_dtype(series):
            parts.append(series.round(precision).astype(str).tolist())
        else:
            parts.append(series.astype(str).str.strip().tolist())
    return pd.Series(["|".join(vals) for vals in zip(*parts)], index=df.index)
