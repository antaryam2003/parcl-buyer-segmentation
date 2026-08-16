"""Phase 5 - encoding and scaling.

The PRD asks for one-hot / label encoding of the categorical fields and for
``StandardScaler`` or ``MinMaxScaler`` on the numeric ones. Rather than
hard-coding one arrangement, this module exposes several *feature sets* and
several *scaler* choices so that :mod:`src.clustering` can select between them
on measured cluster quality instead of on assertion.

Two questions are posed here and answered by measurement rather than by
assertion.

**Does pruning collinear features help?** ``total_area`` correlates with
``total_investment`` at r = 0.98 and ``avg_floor_area`` with
``avg_property_price`` at r = 0.96, because price in this market is almost a
linear function of area (price-per-sqft has a coefficient of variation of
only 5%). Feeding both members of such a pair into a Euclidean distance
doubles the weight of that latent dimension, which is normally a defect.
``behaviour_core`` prunes them and ``behaviour_wide`` keeps them so the
question can be settled empirically. It was: ``behaviour_wide`` wins on all
three internal indices, because the duplicated columns up-weight precisely
the portfolio-scale dimension along which real structure exists.

**Should the categorical fields enter the distance at all?** ``region`` has
57 levels, several with fewer than five clients; after standardisation a
binary column that is non-zero for 0.25% of rows takes the value ~20 for
those rows and dominates the metric. Under min-max scaling the binary flags
sit on the corners of the unit cube and attract centroids irresistibly. The
``prd_encoded`` and ``prd_full`` sets include them so the effect is measured
and reported rather than assumed - see
:func:`src.clustering.flag_redundancy`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer, LabelEncoder, MinMaxScaler, OneHotEncoder,
    StandardScaler,
)

# --------------------------------------------------------------------------
# Column groups
# --------------------------------------------------------------------------

#: Behavioural signals that survive the redundancy prune. ``unique_towers``
#: (r = 0.93 with total_properties) and ``active_months`` (r = 0.85) are
#: dropped as restatements of purchase volume; ``total_area`` and
#: ``avg_floor_area`` are dropped as restatements of the two price columns.
BEHAVIOURAL_CORE = [
    "total_properties",     # purchase volume
    "total_investment",     # capital deployed
    "avg_property_price",   # ticket size
    "price_dispersion",     # ticket consistency
    "office_share",         # commercial vs residential mix
    "purchase_span_days",   # tenure of the relationship
]

#: The literal reading of the PRD's Step-3 list, kept for comparison.
BEHAVIOURAL_WIDE = BEHAVIOURAL_CORE + [
    "avg_floor_area", "total_area", "unique_towers", "active_months",
]

DEMOGRAPHIC_NUMERIC = ["age", "satisfaction_score"]

#: Binary flags are already 0/1 and semantically ordered, so they are scaled
#: with the numerics rather than one-hot expanded.
BINARY_FLAGS = ["is_company", "loan_flag", "is_investment", "is_domestic"]

LOW_CARD_CATEGORICAL = ["gender", "referral_channel"]
HIGH_CARD_CATEGORICAL = ["country", "region"]

#: Right-skewed columns that are log1p-transformed under the ``log`` scaler
#: variant. All are non-negative by construction.
SKEWED = ["total_properties", "total_investment", "price_dispersion"]


@dataclass(frozen=True)
class FeatureSet:
    """A named choice of columns to cluster on."""

    name: str
    description: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def columns(self) -> list[str]:
        return list(self.numeric) + list(self.categorical)


FEATURE_SETS: dict[str, FeatureSet] = {
    "behaviour_core": FeatureSet(
        "behaviour_core",
        "Six de-correlated investment-behaviour features only.",
        tuple(BEHAVIOURAL_CORE), (),
    ),
    "behaviour_wide": FeatureSet(
        "behaviour_wide",
        "Every behavioural feature the PRD lists, including the "
        "collinear area and tower restatements.",
        tuple(BEHAVIOURAL_WIDE), (),
    ),
    "behaviour_demo": FeatureSet(
        "behaviour_demo",
        "Behavioural core plus age and satisfaction.",
        tuple(BEHAVIOURAL_CORE + DEMOGRAPHIC_NUMERIC), (),
    ),
    "behaviour_demo_flags": FeatureSet(
        "behaviour_demo_flags",
        "Behavioural core, demographics and the four binary buyer flags.",
        tuple(BEHAVIOURAL_CORE + DEMOGRAPHIC_NUMERIC + BINARY_FLAGS), (),
    ),
    "prd_encoded": FeatureSet(
        "prd_encoded",
        "Adds one-hot gender, referral channel and country to the above.",
        tuple(BEHAVIOURAL_CORE + DEMOGRAPHIC_NUMERIC + BINARY_FLAGS),
        ("gender", "referral_channel", "country"),
    ),
    "prd_full": FeatureSet(
        "prd_full",
        "Literal PRD encoding: everything above plus one-hot region "
        "(57 levels).",
        tuple(BEHAVIOURAL_CORE + DEMOGRAPHIC_NUMERIC + BINARY_FLAGS),
        ("gender", "referral_channel", "country", "region"),
    ),
}

SCALERS = ("standard", "minmax", "log_standard")


# --------------------------------------------------------------------------
# Matrix construction
# --------------------------------------------------------------------------
def _log1p_frame(X):
    """log1p on a DataFrame/ndarray, preserving shape."""
    return np.log1p(np.asarray(X, dtype=float))


def _numeric_pipeline(scaler: str, columns: list[str]) -> Pipeline:
    steps: list[tuple[str, object]] = []
    if scaler == "log_standard":
        # Only the skewed members of this particular column list.
        mask = [c in SKEWED for c in columns]
        if any(mask):
            idx = [i for i, m in enumerate(mask) if m]

            def _selective_log(X, _idx=idx):
                X = np.asarray(X, dtype=float).copy()
                X[:, _idx] = np.log1p(X[:, _idx])
                return X

            steps.append(("log", FunctionTransformer(
                _selective_log, feature_names_out="one-to-one")))
        steps.append(("scale", StandardScaler()))
    elif scaler == "minmax":
        steps.append(("scale", MinMaxScaler()))
    elif scaler == "standard":
        steps.append(("scale", StandardScaler()))
    else:  # pragma: no cover
        raise ValueError(f"unknown scaler {scaler!r}")
    return Pipeline(steps)


def build_preprocessor(feature_set: FeatureSet,
                       scaler: str = "standard") -> ColumnTransformer:
    """Assemble the encode-and-scale transformer for one feature set."""
    numeric = list(feature_set.numeric)
    transformers: list[tuple] = [
        ("num", _numeric_pipeline(scaler, numeric), numeric),
    ]
    if feature_set.categorical:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                          drop="if_binary"),
            list(feature_set.categorical),
        ))
    return ColumnTransformer(transformers, remainder="drop")


def build_matrix(df: pd.DataFrame, feature_set: FeatureSet,
                 scaler: str = "standard"
                 ) -> tuple[np.ndarray, list[str], ColumnTransformer]:
    """Return the model matrix, its column names and the fitted transformer."""
    pre = build_preprocessor(feature_set, scaler)
    X = pre.fit_transform(df[feature_set.columns])
    names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    return np.asarray(X, dtype=float), names, pre


def label_encode(df: pd.DataFrame,
                 columns: tuple[str, ...] = ("country", "region",
                                             "referral_channel")
                 ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """Integer-encode nominal columns for storage and dashboard indexing.

    The PRD lists label encoding alongside one-hot encoding. Integer codes
    are appropriate for compact storage and for tree-based or lookup use, but
    they impose a false ordering on unordered categories, so these columns
    are deliberately *not* fed into the Euclidean distance used by K-Means.
    """
    out = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in columns:
        if col not in out.columns:
            continue
        enc = LabelEncoder()
        out[f"{col}_code"] = enc.fit_transform(out[col].astype(str))
        encoders[col] = enc
    return out, encoders
