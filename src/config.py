"""Central configuration: paths and analysis constants.

Keeping every path and magic constant in one place makes the pipeline
reproducible and lets the notebooks, the CLI runner and the Streamlit app
all agree on where things live.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_CLIENTS = DATA_DIR / "clients.csv"
RAW_PROPERTIES = DATA_DIR / "properties.csv"

PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_CLIENTS = PROCESSED_DIR / "clients_clean.csv"
CLEAN_PROPERTIES = PROCESSED_DIR / "properties_clean.csv"
CLIENT_FEATURES = PROCESSED_DIR / "client_features.csv"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

SEGMENTED_CLIENTS = OUTPUTS_DIR / "segmented_clients.csv"
CLUSTER_PROFILES = OUTPUTS_DIR / "cluster_profiles.csv"

for _d in (PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Analysis constants
# --------------------------------------------------------------------------

#: Age is measured at the close of the transaction window rather than at
#: "today" so that every re-run of the pipeline reproduces identical ages.
#: 31 December is deliberate: because every client's birthday has already
#: occurred by that date, age reduces to ``2025 - birth_year`` and is
#: therefore immune to the day/month ambiguity documented in
#: :mod:`src.data_cleaning`.
REFERENCE_DATE = pd.Timestamp("2025-12-31")

#: ``clients.date_of_birth`` mixes ``M/D/YYYY`` with ``MM-DD-YYYY``. The
#: slash-separated subset provably uses month-first ordering (its second
#: component ranges 13-31), and ``properties.transaction_date`` is provably
#: month-first as well, so month-first is applied consistently.
DOB_DAYFIRST = False

#: ``properties.transaction_date`` is stored month-first; every transaction
#: is stamped to the first of the month, giving 24 consecutive monthly
#: periods from 2024-01 to 2025-12.
TRANSACTION_DATE_FORMAT = "%m-%d-%Y"

RANDOM_STATE = 42

#: Candidate cluster counts scanned by the elbow / silhouette search.
K_RANGE = range(2, 11)

# --------------------------------------------------------------------------
# Plot styling
# --------------------------------------------------------------------------
FIGSIZE = (10, 6)
DPI = 150

#: Chart surface. Every figure and the dashboard commit to this single light
#: surface: no four-hue set clears the all-pairs colour-vision gates against
#: a dark surface, so one validated light palette is used everywhere and the
#: paper and the dashboard stay visually identical.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"

#: Categorical palette for the four buyer segments. Validated all-pairs in
#: light mode: lightness band PASS, chroma floor PASS, worst colour-vision
#: separation dE 9.2 (aqua/orange, deutan), worst normal-vision separation
#: dE 16.3 (violet/blue). Aqua sits at 2.74:1 against the surface, which
#: triggers the relief rule - every figure using it carries direct labels or
#: an accompanying table, and marker shape is varied as a second channel.
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#4a3aa7",  # violet
]

#: Second encoding channel for scatter plots, paired with PALETTE by index.
MARKERS = ["o", "s", "^", "D"]

#: Canonical reporting order for the four segments, running from the entry
#: tier to the accumulator tier.
SEGMENT_ORDER = [
    "Value-Tier Buyers",
    "Large-Format Premium Buyers",
    "Core Multi-Unit Investors",
    "Portfolio Accumulators",
]

#: Colour is bound to segment *identity*, not to cluster index or rank, so
#: the research paper's figures and the dashboard agree, and filtering a
#: view never repaints the segments that remain.
SEGMENT_COLOURS = dict(zip(SEGMENT_ORDER, PALETTE))
SEGMENT_MARKERS = dict(zip(SEGMENT_ORDER, MARKERS))

#: Single-hue sequential ramp for magnitude encodings (blue, light -> dark).
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
              "#184f95", "#0d366b"]

#: Neutral used where a mark carries no categorical identity.
NEUTRAL = "#6da7ec"
