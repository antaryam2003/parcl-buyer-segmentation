"""Phases 2 and 3 - collapse the transaction table to one row per client.

``properties.csv`` is a transaction ledger: 10,000 listings, of which 7,305
are sold and carry a ``client_ref``. The clustering unit for this project is
the *client*, not the listing, so the ledger is aggregated into a behavioural
profile per client and joined onto the demographic table.

Unsold listings are dropped before aggregation - they describe available
inventory, not buyer behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

#: Engineered behavioural features, grouped for documentation and for the
#: research paper's feature table.
BEHAVIOURAL_FEATURES: dict[str, str] = {
    "total_properties": "Number of units purchased (purchase frequency)",
    "total_investment": "Total capital deployed, USD",
    "avg_property_price": "Mean ticket size per unit, USD",
    "max_property_price": "Largest single purchase, USD",
    "price_dispersion": "Std. dev. of ticket size (0 for single buyers)",
    "avg_floor_area": "Mean unit size, sq ft",
    "total_area": "Total floor area acquired, sq ft",
    "avg_price_per_sqft": "Mean unit price intensity, USD/sq ft",
    "unique_towers": "Distinct towers bought into (diversification)",
    "tower_diversity": "unique_towers / total_properties, 0-1",
    "office_share": "Share of units that are offices, 0-1",
    "apartment_share": "Share of units that are apartments, 0-1",
    "purchase_span_days": "Days between first and last purchase",
    "active_months": "Distinct months with at least one purchase",
    "purchase_intensity": "Purchases per active month",
}

DEMOGRAPHIC_FEATURES: dict[str, str] = {
    "age": "Age in whole years at 31-Dec-2025",
    "satisfaction_score": "Self-reported satisfaction, 1-5",
    "is_company": "1 if client_type is Company",
    "loan_flag": "1 if the client applied for financing",
    "is_investment": "1 if acquisition_purpose is Investment",
    "is_domestic": "1 if country is USA (the home market)",
}


def aggregate_properties(properties: pd.DataFrame) -> pd.DataFrame:
    """Build the per-client behavioural profile from sold transactions."""
    sold = properties.loc[
        properties["listing_status"].eq("Sold")
        & properties["client_ref"].notna()
    ].copy()

    grouped = sold.groupby("client_ref", observed=True)

    agg = grouped.agg(
        total_properties=("listing_id", "count"),
        total_investment=("sale_price", "sum"),
        avg_property_price=("sale_price", "mean"),
        max_property_price=("sale_price", "max"),
        min_property_price=("sale_price", "min"),
        price_dispersion=("sale_price", "std"),
        avg_floor_area=("floor_area_sqft", "mean"),
        total_area=("floor_area_sqft", "sum"),
        avg_price_per_sqft=("price_per_sqft", "mean"),
        unique_towers=("tower_number", "nunique"),
        first_purchase=("transaction_date", "min"),
        last_purchase=("transaction_date", "max"),
        active_months=("transaction_date", "nunique"),
    )

    # std() is undefined for a single observation; a lone purchase has zero
    # observed dispersion, which is the meaningful value here.
    agg["price_dispersion"] = agg["price_dispersion"].fillna(0.0)

    # Unit-mix shares.
    mix = (
        sold.pivot_table(index="client_ref", columns="unit_category",
                         values="listing_id", aggfunc="count",
                         observed=True)
        .fillna(0)
    )
    for col in ("Apartment", "Office"):
        if col not in mix.columns:
            mix[col] = 0
    agg["apartment_count"] = mix["Apartment"].astype(int)
    agg["office_count"] = mix["Office"].astype(int)
    agg["apartment_share"] = agg["apartment_count"] / agg["total_properties"]
    agg["office_share"] = agg["office_count"] / agg["total_properties"]

    # Temporal behaviour.
    agg["purchase_span_days"] = (
        agg["last_purchase"] - agg["first_purchase"]
    ).dt.days.astype(int)
    agg["tower_diversity"] = agg["unique_towers"] / agg["total_properties"]
    agg["purchase_intensity"] = agg["total_properties"] / agg["active_months"]

    agg.index.name = "client_id"
    return agg.reset_index()


def build_client_features(clients: pd.DataFrame, properties: pd.DataFrame,
                          save: bool = True) -> pd.DataFrame:
    """Join demographics to behaviour, producing the analytical base table."""
    behaviour = aggregate_properties(properties)

    keep = [
        "client_id", "full_name", "client_type", "gender", "country",
        "region", "date_of_birth", "age", "acquisition_purpose",
        "satisfaction_score", "loan_applied", "referral_channel",
        "loan_flag", "is_investment", "is_company",
    ]
    df = clients[keep].merge(behaviour, on="client_id", how="left")

    df["is_domestic"] = (df["country"] == "USA").astype(int)
    df["has_purchase"] = df["total_properties"].notna().astype(int)

    # Every client in this extract has at least one purchase, but guard the
    # join anyway so a refreshed extract degrades gracefully instead of
    # emitting NaNs into the scaler.
    behaviour_cols = [c for c in behaviour.columns if c != "client_id"]
    numeric_cols = [c for c in behaviour_cols
                    if not pd.api.types.is_datetime64_any_dtype(df[c])]
    df[numeric_cols] = df[numeric_cols].fillna(0)

    df["satisfaction_score"] = df["satisfaction_score"].astype(int)
    df["age"] = df["age"].astype(int)
    df["total_properties"] = df["total_properties"].astype(int)
    df["unique_towers"] = df["unique_towers"].astype(int)
    df["active_months"] = df["active_months"].astype(int)

    # Convenience bands used by the EDA and the dashboard, never by the model.
    df["age_band"] = pd.cut(
        df["age"], bins=[0, 34, 44, 54, 64, 200],
        labels=["<35", "35-44", "45-54", "55-64", "65+"],
    )
    df["investment_band"] = pd.qcut(
        df["total_investment"], q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
    )

    if save:
        df.to_csv(cfg.CLIENT_FEATURES, index=False)
    return df


def feature_dictionary() -> pd.DataFrame:
    """Tabular data dictionary for the research paper."""
    rows = [("behavioural", k, v) for k, v in BEHAVIOURAL_FEATURES.items()]
    rows += [("demographic", k, v) for k, v in DEMOGRAPHIC_FEATURES.items()]
    return pd.DataFrame(rows, columns=["group", "feature", "description"])


if __name__ == "__main__":  # pragma: no cover
    from .data_cleaning import load_and_clean

    _clients, _props, _ = load_and_clean(save=False)
    _feat = build_client_features(_clients, _props)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)
    print("client feature table:", _feat.shape)
    print(_feat.dtypes.to_string())
    print()
    cols = list(BEHAVIOURAL_FEATURES) + ["age", "satisfaction_score"]
    print(_feat[cols].describe().T.to_string())
    print()
    print("skewness:")
    print(_feat[cols].skew().sort_values(ascending=False).to_string())
    print()
    print("rows with zero purchases:", int((_feat["has_purchase"] == 0).sum()))
    print("total_properties distribution:")
    print(_feat["total_properties"].value_counts().sort_index().to_string())
    print("\nsum check - aggregated purchases vs sold listings:",
          int(_feat["total_properties"].sum()))
    print("sum check - aggregated investment: "
          f"${_feat['total_investment'].sum():,.2f}")
