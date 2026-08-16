"""Phases 8 and 9 - turn cluster indices into named segments and strategy.

Segment names are derived from centroid *ranks*, never hard-coded to a
cluster number, because K-Means label ordering is arbitrary and would
silently scramble the narrative on any re-run.

The naming vocabulary is deliberately restrained. The dataset carries no
income, wealth or net-worth field, so the profiles describe *observed
purchasing behaviour* - capital committed, ticket size, unit footprint,
tenure - and never infer affluence from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

#: Columns summarised for every segment, in reporting order.
PROFILE_NUMERIC = [
    "total_properties", "total_investment", "avg_property_price",
    "max_property_price", "price_dispersion", "avg_floor_area",
    "total_area", "avg_price_per_sqft", "unique_towers", "active_months",
    "purchase_span_days", "office_share", "apartment_share",
    "age", "satisfaction_score",
]

PROFILE_RATES = ["loan_flag", "is_investment", "is_company", "is_domestic"]


def build_profile(df: pd.DataFrame, label_col: str = "cluster"
                  ) -> pd.DataFrame:
    """Mean profile of every cluster plus its share of clients and capital."""
    g = df.groupby(label_col, observed=True)
    prof = g.agg(
        n_clients=("client_id", "size"),
        **{c: (c, "mean") for c in PROFILE_NUMERIC},
        **{c: (c, "mean") for c in PROFILE_RATES},
    )
    prof["pct_clients"] = prof["n_clients"] / len(df)
    prof["capital_committed"] = g["total_investment"].sum()
    prof["pct_capital"] = prof["capital_committed"] / \
        df["total_investment"].sum()
    prof["units_bought"] = g["total_properties"].sum()
    front = ["n_clients", "pct_clients", "capital_committed", "pct_capital",
             "units_bought"]
    return prof[front + PROFILE_NUMERIC + PROFILE_RATES]


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------
#: Descriptions and strategy for the four-segment solution. Keyed by the
#: role the rank rules assign, not by cluster index.
SEGMENT_LIBRARY: dict[str, dict[str, str]] = {
    "Portfolio Accumulators": {
        "thesis": "Highest purchase volume, widest spread across towers and "
                  "the longest active buying window. Two percent of the "
                  "client base, but the largest committed capital per head "
                  "by a wide margin.",
        "signal": "Repeat acquisition behaviour, not a single large ticket.",
        "strategy": "Assign named relationship management. Offer "
                    "portfolio-level pricing, first refusal on new tower "
                    "releases and bulk/blocked allocations. Retention here "
                    "protects a disproportionate share of revenue.",
    },
    "Core Multi-Unit Investors": {
        "thesis": "Above-average unit count at a mid-market ticket, with "
                  "the second-longest buying window. The volume engine of "
                  "the book.",
        "signal": "Steady accumulation at mainstream price points.",
        "strategy": "Nurture toward the accumulator tier with multi-unit "
                    "incentives, staged-purchase plans and loyalty pricing "
                    "on the third and subsequent units.",
    },
    "Large-Format Premium Buyers": {
        "thesis": "Fewest units of any segment, but the largest floor "
                  "areas and the highest price per transaction. Buys "
                  "selectively and concludes quickly.",
        "signal": "Ticket size and unit footprint, not frequency.",
        "strategy": "Lead with penthouse and large-floorplate inventory, "
                    "fit-out and customisation options, and concierge "
                    "handover. Frequency campaigns are wasted here.",
    },
    "Value-Tier Buyers": {
        "thesis": "Smallest units, lowest ticket and the lowest total "
                  "outlay. The widest part of the funnel and the natural "
                  "entry point into the portfolio.",
        "signal": "Price sensitivity and compact unit preference.",
        "strategy": "Pair with financing partners and staged deposit "
                    "plans, promote compact and lower-floor inventory, and "
                    "build an upgrade path so second purchases stay "
                    "in-house.",
    },
}


def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Map cluster ids to segment names using centroid ranks.

    For the four-segment solution the rules are:

    1. the cluster with the highest mean ``total_properties`` is the
       accumulator tier;
    2. of the remainder, the highest mean ``avg_property_price`` is the
       large-format premium tier and the lowest is the value tier;
    3. what is left is the core multi-unit tier.

    For any other K a descriptive fallback name is generated from the
    cluster's position on the capital and ticket-size scales, so the
    function never silently returns a wrong label.
    """
    prof = profile.copy()
    if len(prof) != 4:
        return _fallback_names(prof)

    names: dict[int, str] = {}
    accumulator = prof["total_properties"].idxmax()
    names[accumulator] = "Portfolio Accumulators"

    rest = prof.drop(index=accumulator)
    premium = rest["avg_property_price"].idxmax()
    value = rest["avg_property_price"].idxmin()
    names[premium] = "Large-Format Premium Buyers"
    names[value] = "Value-Tier Buyers"

    for idx in rest.index:
        if idx not in names:
            names[idx] = "Core Multi-Unit Investors"
    return names


def _fallback_names(prof: pd.DataFrame) -> dict[int, str]:
    """Descriptive names for solutions with K != 4."""
    cap = prof["total_investment"].rank(pct=True)
    tic = prof["avg_property_price"].rank(pct=True)
    out = {}
    for idx in prof.index:
        cap_word = ("High" if cap[idx] > 0.66 else
                    "Mid" if cap[idx] > 0.33 else "Low")
        tic_word = ("premium" if tic[idx] > 0.66 else
                    "mid-ticket" if tic[idx] > 0.33 else "value")
        out[idx] = f"{cap_word}-capital {tic_word} buyers"
    return out


def attach_segments(df: pd.DataFrame, labels: np.ndarray,
                    label_col: str = "cluster") -> tuple[pd.DataFrame,
                                                         pd.DataFrame,
                                                         dict[int, str]]:
    """Attach cluster ids and segment names, and return the profile table."""
    out = df.copy()
    out[label_col] = labels
    profile = build_profile(out, label_col)
    names = name_segments(profile)
    out["segment"] = out[label_col].map(names)
    profile.insert(0, "segment", pd.Series(names))
    profile["thesis"] = profile["segment"].map(
        lambda s: SEGMENT_LIBRARY.get(s, {}).get("thesis", ""))
    profile["strategy"] = profile["segment"].map(
        lambda s: SEGMENT_LIBRARY.get(s, {}).get("strategy", ""))
    return out, profile, names


def relative_profile(profile: pd.DataFrame,
                     df: pd.DataFrame) -> pd.DataFrame:
    """Each segment's mean as a ratio to the overall mean (1.0 = average).

    This is the table to read when interpreting clusters: absolute means
    hide which differences are actually large.
    """
    cols = [c for c in PROFILE_NUMERIC + PROFILE_RATES
            if c in profile.columns]
    overall = df[cols].mean()
    rel = profile[cols].div(overall, axis=1)
    rel.insert(0, "segment", profile["segment"])
    return rel


def discriminating_features(profile: pd.DataFrame, df: pd.DataFrame,
                            top_n: int = 6) -> pd.DataFrame:
    """Rank features by how strongly they separate the segments.

    Uses the spread of segment means expressed in overall standard
    deviations, which is comparable across features of different units.
    """
    cols = [c for c in PROFILE_NUMERIC + PROFILE_RATES
            if c in profile.columns]
    sd = df[cols].std().replace(0, np.nan)
    spread = ((profile[cols].max() - profile[cols].min()) / sd)
    return (spread.sort_values(ascending=False)
            .head(top_n)
            .rename("spread_in_sd")
            .rename_axis("feature")
            .reset_index())


def save_outputs(segmented: pd.DataFrame, profile: pd.DataFrame) -> None:
    segmented.to_csv(cfg.SEGMENTED_CLIENTS, index=False)
    profile.to_csv(cfg.CLUSTER_PROFILES)
