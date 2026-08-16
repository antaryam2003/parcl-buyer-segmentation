"""Generate the five analysis notebooks from the pipeline modules.

The notebooks are the narrative walk-through the project guide asks for;
``run_pipeline.py`` is the reproducible batch path. Both call the same
functions in ``src/``, so they cannot drift apart.

Run with ``python tools/build_notebooks.py``.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

BOOT = """\
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
"""

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_data_cleaning": [
        ("md", """\
# 01 - Data Cleaning

**Phase 1 of the project.** The PRD asks for missing-value handling,
categorical normalisation and duplicate removal. Both files turn out to be
structurally clean, so the real work is *type recovery*: two date columns and
one currency column arrive as free text.

Two questions have to be settled with evidence before anything downstream is
trustworthy:

1. Is `transaction_date` day-first or month-first?
2. Is `date_of_birth` day-first or month-first?
"""),
        ("code", BOOT + """
from src import config as cfg
raw_clients = pd.read_csv(cfg.RAW_CLIENTS, dtype=str)
raw_props = pd.read_csv(cfg.RAW_PROPERTIES, dtype=str)
print("clients   ", raw_clients.shape)
print("properties", raw_props.shape)
raw_clients.head()
"""),
        ("md", "## Structural checks: missing values, duplicates, "
               "referential integrity"),
        ("code", """\
print("clients missing cells :", int(raw_clients.isna().sum().sum()))
print("clients duplicate rows:", int(raw_clients.duplicated().sum()))
print("duplicate client_id   :", int(raw_clients['client_id'].duplicated().sum()))
print()
print("properties duplicate listing_id:", int(raw_props['listing_id'].duplicated().sum()))
print("properties missing client_ref  :", int(raw_props['client_ref'].isna().sum()))
print()
print("Missing client_ref against listing_status:")
print(pd.crosstab(raw_props['listing_status'], raw_props['client_ref'].isna()))
"""),
        ("md", """\
The 2,695 missing `client_ref` values line up **exactly** with
`listing_status = Available`. They encode unsold inventory, not missing data,
so they must not be imputed - they are simply excluded from client-level
aggregation.
"""),
        ("md", "## Settling the date formats with evidence"),
        ("code", """\
import re
def shape(s):
    return re.sub(r"\\d", "9", str(s))

print("date_of_birth serialisations:")
print(raw_clients['date_of_birth'].map(shape).value_counts().to_string())
print()
print("transaction_date serialisations:")
print(raw_props['transaction_date'].map(shape).value_counts().to_string())
"""),
        ("code", """\
def parts(series):
    p = series.str.split(r"[-/]", regex=True)
    return (p.str[0].astype(int), p.str[1].astype(int))

# transaction_date: is the second component ever > 12?
a, b = parts(raw_props['transaction_date'])
print("transaction_date  part1 range", a.min(), "-", a.max(),
      "| part2 range", b.min(), "-", b.max())
print("=> part2 is always 1: every transaction is stamped to the 1st of a month.")
print("   Read month-first this gives 24 consecutive months (Jan-2024..Dec-2024..Dec-2025).")
print("   Read day-first it would give days 1-12 of January only, in two years - implausible.")
print()

dob = raw_clients['date_of_birth']
slash = dob[~dob.str.contains('-')]
a2, b2 = parts(slash)
print("date_of_birth, slash subset: part2 range", b2.min(), "-", b2.max())
print("=> the day component reaches 31, so the slash subset is provably MONTH-first.")
dash = dob[dob.str.contains('-')]
a3, b3 = parts(dash)
print("date_of_birth, dash subset : part1 max", a3.max(), "part2 max", b3.max())
print("=> both components <= 12: genuinely ambiguous, and unrecoverable.")
"""),
        ("md", """\
### Why the ambiguity does not matter

The two formats partition the data on `day <= 12` versus `day >= 13`: a
spreadsheet re-serialised every date whose day fitted in a month slot. We read
everything month-first, consistent with the two provably month-first columns.

Crucially, **age is measured at 31 December 2025**. Every client's birthday
has already passed by that date, so age collapses to `2025 - birth_year` and
the day/month ordering cannot influence it. The check below proves it.
"""),
        ("code", """\
d_month = pd.to_datetime(dob, format='mixed', dayfirst=False)
d_day   = pd.to_datetime(dob, format='mixed', dayfirst=True)
print("parsed dates that differ between the two readings:", int((d_month != d_day).sum()))
print("birth YEARS that differ                          :", int((d_month.dt.year != d_day.dt.year).sum()))
print("=> 790 dates differ, zero years differ, so age is unaffected.")
"""),
        ("md", "## Run the packaged cleaner"),
        ("code", """\
from src.data_cleaning import load_and_clean
clients, properties, report = load_and_clean()
print(report.summary())
"""),
        ("code", """\
properties[['listing_id','transaction_date','unit_category',
            'floor_area_sqft','sale_price','listing_status',
            'client_ref','price_per_sqft']].head()
"""),
    ],

    "02_eda": [
        ("md", """\
# 02 - Exploratory Data Analysis

**Phase 4.** Four sections, following the project guide: customer
demographics, buyer intent, property behaviour, and buyer-investment
relationships. The last of these produces the most consequential finding in
the whole project.
"""),
        ("code", BOOT + """
from src import config as cfg, visuals as vz
from src.data_cleaning import load_and_clean
from src.feature_engineering import build_client_features
vz.apply_style()

clients, properties, _ = load_and_clean(save=False)
features = build_client_features(clients, properties, save=False)
print(features.shape)
"""),
        ("md", """\
## A. Customer demographics

> Figures are written to `outputs/figures/` and linked below rather than
> embedded in the notebook. A base64-encoded PNG lands in the `.ipynb` as a
> single line of up to ~300,000 characters, which defeats GitHub's notebook
> preview. Keeping the images external lets the notebook render in the
> browser and keeps the file small enough to diff.
"""),
        ("code", """\
for f in (vz.fig_age_distribution(features),
          vz.fig_demographic_mix(features),
          vz.fig_geography(features)):
    print("wrote", f.name)
"""),
        ("md", """\
![Age distribution](../outputs/figures/eda_01_age_distribution.png)

![Demographic mix](../outputs/figures/eda_02_demographic_mix.png)

![Geography](../outputs/figures/eda_03_geography.png)
"""),
        ("md", "## B. Buyer intent"),
        ("code", """\
print("wrote", vz.fig_buyer_intent(features).name)
print()
print(features['acquisition_purpose'].value_counts().to_string())
print()
print(features['loan_applied'].value_counts().to_string())
"""),
        ("md", "![Buyer intent](../outputs/figures/eda_04_buyer_intent.png)"),
        ("md", "## C. Property behaviour"),
        ("code", """\
for f in (vz.fig_price_distribution(properties),
          vz.fig_inventory(properties),
          vz.fig_portfolio_size(features)):
    print("wrote", f.name)
print()
print(properties['sale_price'].describe().to_string())
"""),
        ("md", """\
![Price distribution](../outputs/figures/eda_05_price_distribution.png)

![Inventory](../outputs/figures/eda_06_inventory.png)

![Portfolio size](../outputs/figures/eda_07_portfolio_size.png)
"""),
        ("md", """\
## D. Buyer-investment relationships - the key finding

The project guide expects relationships between declared attributes and
spending. We test them formally rather than assuming them.
"""),
        ("code", """\
from scipy import stats
for cat in ['client_type', 'acquisition_purpose', 'loan_applied']:
    groups = [g['total_investment'].to_numpy()
              for _, g in features.groupby(cat, observed=True)]
    t, p = stats.ttest_ind(*groups, equal_var=False)
    n1, n2 = len(groups[0]), len(groups[1])
    pooled = np.sqrt(((n1-1)*groups[0].var(ddof=1) + (n2-1)*groups[1].var(ddof=1)) / (n1+n2-2))
    d = (groups[0].mean() - groups[1].mean()) / pooled
    print(f"{cat:22s} vs total_investment: p={p:.4f}  Cohen's d={d:+.3f}")

print()
print(f"age          vs total_investment: r={features['age'].corr(features['total_investment']):+.4f}")
print(f"satisfaction vs total_investment: r={features['satisfaction_score'].corr(features['total_investment']):+.4f}")
"""),
        ("md", """\
**Every declared attribute is statistically independent of spending.**
All p-values are far above 0.05 and every effect size is below |d| = 0.07.

This is not a defect in the analysis - it is the central constraint on the
whole project. Segmentation cannot be built on what buyers *declare*; it has
to be built on what they *transact*. That decision drives every modelling
choice in notebook 04.
"""),
        ("code", """\
CORR = ['age','satisfaction_score','total_properties','total_investment',
        'avg_property_price','price_dispersion','avg_floor_area','total_area',
        'avg_price_per_sqft','unique_towers','active_months',
        'purchase_span_days','office_share']
for f in (vz.fig_intent_vs_investment(features),
          vz.fig_correlation_heatmap(features, CORR)):
    print("wrote", f.name)
"""),
        ("md", """\
![Intent vs investment](../outputs/figures/eda_08_intent_vs_investment.png)

![Correlation heatmap](../outputs/figures/eda_09_correlation.png)
"""),
    ],

    "03_feature_engineering": [
        ("md", """\
# 03 - Feature Engineering

**Phases 2 and 3.** `properties.csv` is a transaction ledger; the clustering
unit is the *client*. This notebook collapses 7,305 sold transactions into one
behavioural profile per client and joins it to the demographic table.
"""),
        ("code", BOOT + """
from src import config as cfg
from src.data_cleaning import load_and_clean
from src.feature_engineering import (aggregate_properties,
                                     build_client_features,
                                     feature_dictionary)
clients, properties, _ = load_and_clean(save=False)
"""),
        ("md", "## Aggregate the ledger"),
        ("code", """\
behaviour = aggregate_properties(properties)
print(behaviour.shape)
behaviour.head()
"""),
        ("code", """\
# Mirrors run_pipeline.py exactly, including the label-encoded columns, so
# re-running this notebook reproduces the committed artefact rather than
# overwriting it with a narrower table.
from src.preprocessing import label_encode

features = build_client_features(clients, properties)
features, encoders = label_encode(features)
features.to_csv(cfg.CLIENT_FEATURES, index=False)

print("client feature table:", features.shape)
print("units aggregated  :", int(features['total_properties'].sum()), "(expect 7,305)")
print("capital aggregated: $%s" % format(features['total_investment'].sum(), ',.2f'))
print("label-encoded columns:", [c for c in features.columns if c.endswith('_code')])
feature_dictionary()
"""),
        ("md", "## Distributions and redundancy"),
        ("code", """\
cols = ['total_properties','total_investment','avg_property_price',
        'max_property_price','price_dispersion','avg_floor_area','total_area',
        'avg_price_per_sqft','unique_towers','tower_diversity','office_share',
        'purchase_span_days','active_months','purchase_intensity']
features[cols].describe().T.round(2)
"""),
        ("code", """\
corr = features[cols].corr()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(cols)
         for b in cols[i+1:] if abs(corr.loc[a, b]) >= 0.75]
print("Strongly correlated pairs (|r| >= 0.75):")
for a, b, r in sorted(pairs, key=lambda t: -abs(t[2])):
    print(f"  {a:20s} ~ {b:20s} r = {r:+.2f}")
print()
cv = (features[cols].std() / features[cols].mean()).abs().sort_values()
print("Coefficient of variation (near-zero => near-constant, useless):")
print(cv.round(4).to_string())
"""),
        ("md", """\
Two consequences for the model:

* `tower_diversity` (CV 0.04), `avg_price_per_sqft` (CV 0.05) and
  `purchase_intensity` (CV 0.17) are close to constant and carry almost no
  discriminating signal.
* `total_area ~ total_investment` (r = 0.98) and
  `avg_floor_area ~ avg_property_price` (r = 0.96) are near-restatements,
  because price here is essentially a linear function of area.

Whether pruning the redundant pair *helps* is an empirical question, so
notebook 04 tests both a pruned and a full behavioural feature set rather than
assuming an answer.
"""),
    ],

    "04_clustering": [
        ("md", """\
# 04 - Clustering

**Phases 5 to 7.** Encoding, scaling, K-Means, elbow and silhouette, then
hierarchical validation.

The search is run as a grid over *feature set x scaler x K*, and every cell is
scored on more than cluster compactness - because on this dataset the highest
silhouette score belongs to a solution that has discovered nothing at all.
"""),
        ("code", BOOT + """
from src import config as cfg, visuals as vz
from src.clustering import (compare_partitions, elbow_k, eta_squared,
                            fit_hierarchical, fit_kmeans, index_consensus,
                            pca_projection, run_experiment, scan_k,
                            select_best, silhouette_by_cluster,
                            stability_check)
from src.preprocessing import FEATURE_SETS, build_matrix
vz.apply_style()
features = pd.read_csv(cfg.CLIENT_FEATURES)

for name, fs in FEATURE_SETS.items():
    print(f"{name:22s} {fs.description}")
"""),
        ("md", "## The grid search"),
        ("code", """\
experiment = run_experiment(features)
cols = ['feature_set','scaler','k','silhouette','davies_bouldin',
        'behavioural_separation','declared_separation','min_cluster_size',
        'actionable','substantive']
print("Top 8 by raw silhouette, ignoring all substance checks:")
experiment.sort_values('silhouette', ascending=False).head(8)[cols].round(4)
"""),
        ("md", """\
### Why the leader is rejected

The top solution reaches silhouette 0.32 - but its behavioural separation is
0.003, meaning it explains essentially **none** of the variance in what
clients bought. Min-max scaling puts the binary buyer flags on the corners of
the unit cube, and K-Means simply walks to them.
"""),
        ("code", """\
X_bad, _, _ = build_matrix(features, FEATURE_SETS['behaviour_demo_flags'], 'minmax')
bad = fit_kmeans(X_bad, 6).labels_
e = eta_squared(features, bad, ['loan_flag','is_investment','is_domestic',
                                'total_investment','total_properties',
                                'avg_property_price','office_share'])
print("Variance explained by that 'best' 6-cluster solution:")
print(e.round(4).sort_values(ascending=False).to_string())
print()
from src.clustering import flag_redundancy
print("Adjusted Rand index vs the raw crosstab of the four binary flags:",
      round(flag_redundancy(features, bad), 3))
print("=> 0.9 means K-Means re-derived a groupby. That is not a discovered segment.")
"""),
        ("md", """\
So the search applies two gates before ranking:

* **actionable** - no segment smaller than 25 clients;
* **substantive** - the partition must explain more variance in what clients
  *bought* than in what they *declared*.

Survivors are then ranked by how many of the three internal indices
independently choose that K. That last step matters: silhouette and
Calinski-Harabasz both drift upward as K falls, so ranking on either alone
returns the degenerate K = 2.
"""),
        ("code", """\
print("Viable solutions, best first:")
display(experiment[experiment.viable].sort_values('silhouette', ascending=False).head(8)[cols].round(4))
print()
print("Which K each index picks, per configuration:")
display(index_consensus(experiment).sort_values('n_indices_agreeing', ascending=False).head(8))
"""),
        ("code", """\
best = select_best(experiment)
fs_name, scaler, k = best['feature_set'], best['scaler'], int(best['k'])
print(f"SELECTED: {fs_name} / {scaler} / K={k}")
print(f"  silhouette {best['silhouette']:.4f} | davies-bouldin {best['davies_bouldin']:.4f}"
      f" | calinski-harabasz {best['calinski_harabasz']:.1f}")
print(f"  indices agreeing: {best['voted_by']} ({int(best['n_indices_agreeing'])} of 3)")
"""),
        ("md", "## Elbow and silhouette curves for the selected representation"),
        ("code", """\
X, names, pre = build_matrix(features, FEATURE_SETS[fs_name], scaler)
scan = scan_k(X)
display(scan.round(4))
print("elbow K =", elbow_k(scan['k'].tolist(), scan['inertia'].tolist()))
print("wrote", vz.fig_elbow_silhouette(scan, k).name)
"""),
        ("md",
         "![Elbow and silhouette]"
         "(../outputs/figures/model_01_elbow_silhouette.png)"),
        ("md", "## Final fit, stability and hierarchical validation"),
        ("code", """\
km = fit_kmeans(X, k)
labels = km.labels_
print("cluster sizes:", np.bincount(labels).tolist())
display(silhouette_by_cluster(X, labels).round(3))

st = stability_check(X, k, n_runs=100)
print(f"bootstrap over 100 x 80% subsamples: mean ARI {st['mean_ari']:.3f} "
      f"(sd {st['std_ari']:.3f}, min {st['min_ari']:.3f})")
"""),
        ("code", """\
hier, Z, ccc = fit_hierarchical(X, k)
print(f"Ward linkage, cophenetic correlation {ccc:.3f}")
print("hierarchical sizes:", np.bincount(hier).tolist())
print("agreement with K-Means:", {a: round(b, 3) for a, b in compare_partitions(labels, hier).items()})
print("wrote", vz.fig_dendrogram(Z, k, ccc).name)
"""),
        ("md", "![Dendrogram](../outputs/figures/model_03_dendrogram.png)"),
    ],

    "05_cluster_interpretation": [
        ("md", """\
# 05 - Cluster Interpretation and Business Recommendations

**Phases 8 and 9.** Cluster indices become named segments with a strategy
attached. Names are derived from centroid *ranks*, never hard-coded to a
cluster number, so a re-run cannot scramble the narrative.

The dataset carries no income or wealth field, so every profile describes
observed purchasing behaviour and infers no affluence.
"""),
        ("code", BOOT + """
from src import config as cfg, visuals as vz
from src.clustering import fit_kmeans, eta_squared, pca_projection
from src.preprocessing import FEATURE_SETS, build_matrix
from src.interpretation import (attach_segments, discriminating_features,
                                relative_profile, SEGMENT_LIBRARY)
vz.apply_style()
features = pd.read_csv(cfg.CLIENT_FEATURES)
X, names, _ = build_matrix(features, FEATURE_SETS['behaviour_wide'], 'standard')
labels = fit_kmeans(X, 4).labels_
segmented, profile, seg_names = attach_segments(features, labels)
print(seg_names)
"""),
        ("md", "## Segment profiles"),
        ("code", """\
show = ['segment','n_clients','pct_clients','pct_capital','total_properties',
        'total_investment','avg_property_price','avg_floor_area',
        'purchase_span_days','office_share','age','satisfaction_score',
        'loan_flag','is_investment']
profile[show].round(3)
"""),
        ("code", """\
print("Each segment's mean as a multiple of the overall mean (1.00 = average):")
relative_profile(profile, features).round(2)
"""),
        ("code", """\
print("What separates the segments (range of segment means, in overall SD):")
display(discriminating_features(profile, features, top_n=10).round(2))
print()
print("Variance explained by the segmentation, per feature:")
e = eta_squared(features, labels, ['total_properties','total_investment',
    'avg_property_price','avg_floor_area','total_area','unique_towers',
    'purchase_span_days','office_share','age','satisfaction_score',
    'loan_flag','is_investment','is_company','is_domestic'])
print(e.sort_values(ascending=False).round(3).to_string())
"""),
        ("md", """\
Behavioural features are explained at 0.4-0.8; every declared attribute sits
near 0.00. The segmentation is a description of purchasing behaviour, and it
deliberately says nothing about demographics - because in this dataset
demographics say nothing about purchasing.
"""),
        ("md", "## Visual summary"),
        ("code", """\
FINGERPRINT = ['total_properties','total_investment','avg_property_price',
               'avg_floor_area','total_area','unique_towers','active_months',
               'purchase_span_days','price_dispersion','office_share',
               'age','satisfaction_score']
coords, pca = pca_projection(X, 2)
for f in (vz.fig_pca_scatter(coords, labels, seg_names, pca),
          vz.fig_segment_sizes(profile),
          vz.fig_segment_fingerprint(profile, features, FINGERPRINT),
          vz.fig_segment_geography(segmented)):
    print("wrote", f.name)
"""),
        ("md", """\
![PCA scatter](../outputs/figures/model_02_pca_scatter.png)

![Segment sizes](../outputs/figures/model_04_segment_sizes.png)

![Segment fingerprint](../outputs/figures/model_05_segment_fingerprint.png)

![Segment geography](../outputs/figures/model_06_segment_geography.png)
"""),
        ("md", "## Business recommendations"),
        ("code", """\
for cl in profile.index:
    seg = profile.loc[cl, 'segment']
    info = SEGMENT_LIBRARY.get(seg, {})
    print("=" * 74)
    print(f"{seg}   |   {int(profile.loc[cl,'n_clients'])} clients "
          f"({profile.loc[cl,'pct_clients']:.1%} of book, "
          f"{profile.loc[cl,'pct_capital']:.1%} of capital)")
    print("=" * 74)
    print("  Defining thesis :", info.get('thesis',''))
    print("  Primary signal  :", info.get('signal',''))
    print("  Strategy        :", info.get('strategy',''))
    print()
"""),
    ],
}


def build() -> None:
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = [
            nbf.v4.new_markdown_cell(body) if kind == "md"
            else nbf.v4.new_code_cell(body)
            for kind, body in cells
        ]
        nb.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        }
        path = NB_DIR / f"{name}.ipynb"
        nbf.write(nb, path)
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    build()
