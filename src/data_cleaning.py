"""Phase 1 - load and clean ``clients.csv`` and ``properties.csv``.

The PRD asks for missing-value handling, categorical normalisation and
duplicate removal. Both files turn out to be structurally clean, so the real
work here is *type recovery*: two date columns and one currency column arrive
as free text and have to be parsed correctly before anything downstream can
use them.

Every decision that could plausibly have gone another way is recorded in
:func:`build_cleaning_report` so the research paper can cite it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import config as cfg

# Categorical columns whose labels are normalised (trimmed) on load.
_CLIENT_CATEGORICALS = [
    "client_type", "gender", "country", "region",
    "acquisition_purpose", "loan_applied", "referral_channel",
]
_PROPERTY_CATEGORICALS = ["unit_category", "listing_status"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def parse_currency(series: pd.Series) -> pd.Series:
    """Turn ``"$300,385.62"`` into ``300385.62``.

    Anything that cannot be parsed becomes ``NaN`` rather than raising, so a
    single malformed row cannot abort the pipeline; the caller checks the
    resulting null count.
    """
    cleaned = (
        series.astype("string")
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalise_labels(df: pd.DataFrame, columns: list[str]) -> int:
    """Strip surrounding whitespace and collapse internal runs of spaces.

    Returns the number of cell values that were actually altered, which the
    cleaning report quotes.
    """
    changed = 0
    for col in columns:
        if col not in df.columns:
            continue
        original = df[col].astype("string")
        tidy = original.str.strip().str.replace(r"\s+", " ", regex=True)
        changed += int((original.fillna("") != tidy.fillna("")).sum())
        df[col] = tidy
    return changed


# --------------------------------------------------------------------------
# Cleaning report
# --------------------------------------------------------------------------
@dataclass
class CleaningReport:
    """Structured audit trail of everything the cleaning step did."""

    clients_rows_in: int = 0
    clients_rows_out: int = 0
    clients_missing_cells: int = 0
    clients_duplicate_rows: int = 0
    clients_duplicate_ids: int = 0
    properties_rows_in: int = 0
    properties_rows_out: int = 0
    properties_duplicate_rows: int = 0
    properties_duplicate_ids: int = 0
    properties_missing_client_ref: int = 0
    missing_ref_all_available: bool = False
    orphan_client_refs: int = 0
    clients_with_purchase: int = 0
    labels_normalised: int = 0
    dob_unparsed: int = 0
    dob_ambiguous_rows: int = 0
    price_unparsed: int = 0
    transaction_min: Any = None
    transaction_max: Any = None
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = [(k, v) for k, v in self.__dict__.items() if k != "notes"]
        return pd.DataFrame(rows, columns=["check", "value"])

    def summary(self) -> str:
        lines = [
            "=" * 68,
            "DATA CLEANING REPORT",
            "=" * 68,
            f"clients      : {self.clients_rows_in} in -> "
            f"{self.clients_rows_out} out",
            f"  missing cells      : {self.clients_missing_cells}",
            f"  duplicate rows     : {self.clients_duplicate_rows}",
            f"  duplicate client_id: {self.clients_duplicate_ids}",
            f"  DOB unparsed       : {self.dob_unparsed}",
            f"  DOB day/month ambiguous rows: {self.dob_ambiguous_rows}",
            f"properties   : {self.properties_rows_in} in -> "
            f"{self.properties_rows_out} out",
            f"  duplicate rows     : {self.properties_duplicate_rows}",
            f"  duplicate listing_id: {self.properties_duplicate_ids}",
            f"  missing client_ref : {self.properties_missing_client_ref}",
            f"  all missing refs are 'Available': "
            f"{self.missing_ref_all_available}",
            f"  orphan client_ref  : {self.orphan_client_refs}",
            f"  sale_price unparsed: {self.price_unparsed}",
            f"  transaction window : {self.transaction_min} .. "
            f"{self.transaction_max}",
            f"linkage      : {self.clients_with_purchase} of "
            f"{self.clients_rows_out} clients have >=1 sold property",
            f"labels normalised (cells changed): {self.labels_normalised}",
        ]
        if self.notes:
            lines.append("-" * 68)
            lines.extend(f"note: {n}" for n in self.notes)
        lines.append("=" * 68)
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Cleaners
# --------------------------------------------------------------------------
def clean_clients(raw: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Normalise client labels, parse dates and derive ``age``."""
    df = raw.copy()
    report.clients_rows_in = len(df)
    report.clients_missing_cells = int(df.isna().sum().sum())
    report.clients_duplicate_rows = int(df.duplicated().sum())
    report.clients_duplicate_ids = int(df["client_id"].duplicated().sum())

    # Duplicate removal is requested by the PRD. It is a no-op on this
    # extract but we run it so the pipeline stays correct if the data is
    # ever refreshed.
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="client_id", keep="first")

    report.labels_normalised += _normalise_labels(df, _CLIENT_CATEGORICALS)

    # ---- date_of_birth -------------------------------------------------
    # Two serialisations are present. The slash subset is provably
    # month-first (its day component reaches 31); the dash subset has both
    # components <= 12 and is therefore genuinely ambiguous. We read
    # everything month-first for consistency with the slash subset and with
    # properties.transaction_date. Because age is measured on 31 December,
    # only the birth *year* matters, so the ambiguity cannot propagate.
    dob_raw = df["date_of_birth"].astype("string")
    dash = dob_raw.str.contains("-", na=False)
    parts = dob_raw.str.split(r"[-/]", regex=True)
    both_small = parts.map(
        lambda p: isinstance(p, list) and len(p) == 3
        and int(p[0]) <= 12 and int(p[1]) <= 12
    )
    report.dob_ambiguous_rows = int((dash & both_small).sum())

    dob = pd.to_datetime(dob_raw, format="mixed",
                         dayfirst=cfg.DOB_DAYFIRST, errors="coerce")
    report.dob_unparsed = int(dob.isna().sum())
    df["date_of_birth"] = dob

    # Whole years elapsed at the reference date.
    ref = cfg.REFERENCE_DATE
    had_birthday = (
        (dob.dt.month < ref.month)
        | ((dob.dt.month == ref.month) & (dob.dt.day <= ref.day))
    )
    df["age"] = (ref.year - dob.dt.year - (~had_birthday).astype(int))
    df["age"] = df["age"].astype("Int64")

    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"],
                                             errors="coerce").astype("Int64")
    # Binary flag alongside the label, so it can be averaged per cluster.
    df["loan_flag"] = (df["loan_applied"].str.lower() == "yes").astype(int)
    df["is_investment"] = (
        df["acquisition_purpose"].str.lower() == "investment"
    ).astype(int)
    df["is_company"] = (df["client_type"].str.lower() == "company").astype(int)
    df["full_name"] = df["first_name"].str.strip() + " " + \
        df["last_name"].str.strip()

    report.clients_rows_out = len(df)
    return df.reset_index(drop=True)


def clean_properties(raw: pd.DataFrame,
                     report: CleaningReport) -> pd.DataFrame:
    """Parse the transaction date and the currency column, coerce numerics."""
    df = raw.copy()
    report.properties_rows_in = len(df)
    report.properties_duplicate_rows = int(df.duplicated().sum())
    report.properties_duplicate_ids = int(df["listing_id"].duplicated().sum())

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="listing_id", keep="first")

    report.labels_normalised += _normalise_labels(df, _PROPERTY_CATEGORICALS)

    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"], format=cfg.TRANSACTION_DATE_FORMAT,
        errors="coerce",
    )
    report.transaction_min = df["transaction_date"].min().date()
    report.transaction_max = df["transaction_date"].max().date()

    df["sale_price"] = parse_currency(df["sale_price"])
    report.price_unparsed = int(df["sale_price"].isna().sum())

    for col in ("tower_number", "unit_number"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["floor_area_sqft"] = pd.to_numeric(df["floor_area_sqft"],
                                          errors="coerce")
    df["listing_id"] = pd.to_numeric(df["listing_id"],
                                     errors="coerce").astype("Int64")

    # client_ref is legitimately null for unsold stock - do NOT impute.
    df["client_ref"] = df["client_ref"].astype("string").str.strip()
    df.loc[df["client_ref"].isin(["", "nan", "None"]), "client_ref"] = pd.NA

    missing_ref = df["client_ref"].isna()
    report.properties_missing_client_ref = int(missing_ref.sum())
    report.missing_ref_all_available = bool(
        (df.loc[missing_ref, "listing_status"] == "Available").all()
        and (df.loc[~missing_ref, "listing_status"] == "Sold").all()
    )

    # price_per_sqft is cheap to derive here and useful in both EDA and
    # feature engineering.
    df["price_per_sqft"] = df["sale_price"] / df["floor_area_sqft"]

    report.properties_rows_out = len(df)
    return df.reset_index(drop=True)


def check_linkage(clients: pd.DataFrame, properties: pd.DataFrame,
                  report: CleaningReport) -> None:
    """Validate the ``client_id`` -> ``client_ref`` foreign key."""
    known = set(clients["client_id"])
    refs = properties["client_ref"].dropna()
    report.orphan_client_refs = int((~refs.isin(known)).sum())
    report.clients_with_purchase = int(len(set(refs) & known))

    if report.orphan_client_refs == 0:
        report.notes.append(
            "Referential integrity holds: every client_ref resolves to a "
            "client_id."
        )
    if report.clients_with_purchase == len(known):
        report.notes.append(
            "All clients appear in the sold-property table, so behavioural "
            "features are available for the entire client base."
        )
    if report.missing_ref_all_available:
        report.notes.append(
            "Missing client_ref values coincide exactly with "
            "listing_status='Available'; they encode unsold stock, not "
            "missing data, and are excluded from client aggregation rather "
            "than imputed."
        )
    if report.dob_ambiguous_rows:
        report.notes.append(
            f"{report.dob_ambiguous_rows} date_of_birth values have both "
            "leading components <= 12 and are day/month ambiguous. Age is "
            "measured on 31-Dec-2025, where age reduces to "
            "(2025 - birth_year), so the ambiguity has no effect on any "
            "downstream feature."
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def load_and_clean(save: bool = True
                   ) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Run the full Phase-1 clean and optionally persist the outputs."""
    report = CleaningReport()

    raw_clients = pd.read_csv(cfg.RAW_CLIENTS, dtype=str)
    raw_properties = pd.read_csv(cfg.RAW_PROPERTIES, dtype=str)

    clients = clean_clients(raw_clients, report)
    properties = clean_properties(raw_properties, report)
    check_linkage(clients, properties, report)

    if save:
        clients.to_csv(cfg.CLEAN_CLIENTS, index=False)
        properties.to_csv(cfg.CLEAN_PROPERTIES, index=False)
        report.to_frame().to_csv(cfg.TABLES_DIR / "cleaning_report.csv",
                                 index=False)
    return clients, properties, report


if __name__ == "__main__":  # pragma: no cover
    _c, _p, _r = load_and_clean()
    print(_r.summary())
