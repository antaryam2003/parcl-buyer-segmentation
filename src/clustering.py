"""Phases 6 and 7 - K-Means, cluster-count selection, hierarchical validation.

Selection is deliberately not left to a single statistic. For every
(feature set, scaler, K) combination the module records inertia, silhouette,
Calinski-Harabasz and Davies-Bouldin, plus the size of the smallest cluster,
because a high silhouette bought by isolating a nine-client singleton is not
a usable business segment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score,
    calinski_harabasz_score, davies_bouldin_score, silhouette_samples,
    silhouette_score,
)

from . import config as cfg
from .preprocessing import FEATURE_SETS, SCALERS, FeatureSet, build_matrix


# --------------------------------------------------------------------------
# K selection
# --------------------------------------------------------------------------
def elbow_k(ks: list[int], inertia: list[float]) -> int:
    """Locate the elbow as the point furthest from the first-to-last chord.

    This is the standard geometric reading of an elbow plot and removes the
    subjectivity of eyeballing the curve.
    """
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertia, dtype=float)
    x = (x - x.min()) / np.ptp(x)
    y = (y - y.min()) / np.ptp(y)
    p0, p1 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    chord = p1 - p0
    chord = chord / np.linalg.norm(chord)
    pts = np.column_stack([x, y]) - p0
    proj = np.outer(pts @ chord, chord)
    dist = np.linalg.norm(pts - proj, axis=1)
    return int(ks[int(np.argmax(dist))])


def scan_k(X: np.ndarray, k_range=cfg.K_RANGE,
           random_state: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """Fit K-Means for each K and score it four different ways."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X)
        sizes = np.bincount(labels, minlength=k)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "min_cluster_size": int(sizes.min()),
            "min_cluster_pct": float(sizes.min() / len(labels)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Model-selection experiment
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    feature_set: str
    scaler: str
    k: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    inertia: float
    min_cluster_pct: float
    n_features: int


#: Behavioural columns used to test whether a partition actually separates
#: investment behaviour, as opposed to re-deriving a categorical crosstab.
_SEPARATION_FEATURES = [
    "total_properties", "total_investment", "avg_property_price",
    "price_dispersion", "office_share", "purchase_span_days",
]

#: Categorical buyer attributes a partition must not merely reproduce.
_TRIVIAL_FLAGS = ["is_company", "loan_flag", "is_investment", "is_domestic"]

#: Declared/demographic attributes. These are low-cardinality and, in this
#: dataset, statistically independent of purchasing behaviour, so a clustering
#: that separates mainly on them has re-derived a filter rather than
#: discovered a segment.
_DECLARED_FEATURES = ["age", "satisfaction_score", "is_company", "loan_flag",
                      "is_investment", "is_domestic"]


def eta_squared(df: pd.DataFrame, labels: np.ndarray,
                columns: list[str] | None = None) -> pd.Series:
    """Share of each column's variance explained by the cluster assignment.

    eta^2 = SS_between / SS_total. It answers "how much of the spread in
    this feature does the segmentation actually account for?", which is the
    question a silhouette score cannot answer.
    """
    columns = columns or _SEPARATION_FEATURES
    out = {}
    tmp = df.assign(_cl=labels)
    for col in columns:
        grand = tmp[col].mean()
        ss_tot = ((tmp[col] - grand) ** 2).sum()
        if ss_tot == 0:
            out[col] = 0.0
            continue
        ss_bet = sum(len(g) * (g[col].mean() - grand) ** 2
                     for _, g in tmp.groupby("_cl", observed=True))
        out[col] = float(ss_bet / ss_tot)
    return pd.Series(out)


def behavioural_separation(df: pd.DataFrame, labels: np.ndarray) -> float:
    """Mean eta^2 across the behavioural features - the 'is this real?' score."""
    return float(eta_squared(df, labels, _SEPARATION_FEATURES).mean())


def declared_separation(df: pd.DataFrame, labels: np.ndarray) -> float:
    """Mean eta^2 across the declared demographic and intent attributes."""
    return float(eta_squared(df, labels, _DECLARED_FEATURES).mean())


def flag_redundancy(df: pd.DataFrame, labels: np.ndarray) -> float:
    """Adjusted Rand index against the raw crosstab of the binary flags.

    A value near 1 means K-Means has simply re-derived a grouping that a
    two-line ``groupby`` already provides - statistically tidy, but not a
    discovered segmentation.
    """
    combo = df[_TRIVIAL_FLAGS].astype(int).astype(str).agg("".join, axis=1)
    return float(adjusted_rand_score(combo, labels))


def run_experiment(df: pd.DataFrame,
                   feature_sets: dict[str, FeatureSet] | None = None,
                   scalers: tuple[str, ...] = SCALERS,
                   k_range=cfg.K_RANGE,
                   min_cluster_size: int = 25,
                   min_behavioural_separation: float = 0.10,
                   random_state: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """Grid-search feature set x scaler x K, scoring quality *and* substance.

    Alongside the four internal validity indices, each cell records the
    behavioural separation it achieves and how far it merely restates the
    binary buyer flags, so that degenerate solutions can be rejected on
    stated grounds rather than quietly dropped.
    """
    feature_sets = feature_sets or FEATURE_SETS
    records: list[dict] = []
    for fs_name, fs in feature_sets.items():
        for scaler in scalers:
            X, _names, _ = build_matrix(df, fs, scaler)
            for k in k_range:
                km = KMeans(n_clusters=k, n_init=20,
                            random_state=random_state)
                labels = km.fit_predict(X)
                sizes = np.bincount(labels, minlength=k)
                records.append({
                    "feature_set": fs_name,
                    "scaler": scaler,
                    "n_features": X.shape[1],
                    "k": k,
                    "inertia": km.inertia_,
                    "silhouette": silhouette_score(X, labels),
                    "calinski_harabasz": calinski_harabasz_score(X, labels),
                    "davies_bouldin": davies_bouldin_score(X, labels),
                    "min_cluster_size": int(sizes.min()),
                    "min_cluster_pct": float(sizes.min() / len(labels)),
                    "behavioural_separation":
                        behavioural_separation(df, labels),
                    "declared_separation": declared_separation(df, labels),
                    "flag_redundancy": flag_redundancy(df, labels),
                })

    out = pd.DataFrame(records)
    out["actionable"] = out["min_cluster_size"] >= min_cluster_size
    # A buyer *investment* segmentation must explain more of the variance in
    # what clients bought than in what they declared on a form. The
    # comparison is threshold-free; the small absolute floor only rules out
    # the degenerate case where a partition explains nothing at all.
    out["substantive"] = (
        (out["behavioural_separation"] > out["declared_separation"])
        & (out["behavioural_separation"] >= min_behavioural_separation)
    )
    out["viable"] = out["actionable"] & out["substantive"]
    return out


def index_consensus(experiment: pd.DataFrame) -> pd.DataFrame:
    """For each configuration, count how many indices agree on each K.

    Silhouette and Calinski-Harabasz both drift upward as K falls, so
    ranking on silhouette alone reliably returns K = 2 - a split into
    "large" and "small" that is statistically tidy and commercially inert.
    Davies-Bouldin does not share that bias. Requiring the three to *agree*
    removes the bias without hand-picking a winner: a configuration where
    all three independently land on the same K is far stronger evidence
    than a marginally higher score from any one of them.
    """
    rows = []
    for (fs, sc), g in experiment.groupby(["feature_set", "scaler"]):
        votes = {
            "silhouette": int(g.loc[g["silhouette"].idxmax(), "k"]),
            "davies_bouldin": int(g.loc[g["davies_bouldin"].idxmin(), "k"]),
            "calinski_harabasz":
                int(g.loc[g["calinski_harabasz"].idxmax(), "k"]),
        }
        for k, count in pd.Series(list(votes.values())).value_counts().items():
            rows.append({
                "feature_set": fs, "scaler": sc, "k": int(k),
                "n_indices_agreeing": int(count),
                "voted_by": ",".join(sorted(name for name, kk in votes.items()
                                            if kk == k)),
            })
    return pd.DataFrame(rows)


def select_best(experiment: pd.DataFrame) -> pd.Series:
    """Choose the configuration with the strongest, most usable evidence.

    Two gates are applied before ranking:

    * **actionable** - every segment holds at least ``min_cluster_size``
      clients, so it can carry a campaign. This is an absolute floor, not a
      percentage: a 39-client segment averaging $2.6M of committed capital
      is small but commercially significant, and a percentage floor would
      discard exactly the high-value tail the project exists to find.
    * **substantive** - the partition explains more variance in what
      clients *bought* than in what they *declared*. Min-max scaling places
      binary flags on the corners of the unit cube and stretches a 1-5
      satisfaction score across the full range, so K-Means will chase those
      low-cardinality columns to a high silhouette while explaining none of
      the investment behaviour. Both failure modes are caught by the same
      comparison, which needs no arbitrary cut-off.

    Survivors are then ranked by how many of the three internal indices
    independently select that K (see :func:`index_consensus`), and only then
    by silhouette. This is what stops the search from settling on K = 2.
    """
    viable = experiment[experiment["viable"]]
    if viable.empty:  # pragma: no cover
        viable = experiment

    consensus = index_consensus(experiment)
    ranked = viable.merge(consensus, on=["feature_set", "scaler", "k"],
                          how="left")
    ranked["n_indices_agreeing"] = ranked["n_indices_agreeing"].fillna(0)
    ranked["voted_by"] = ranked["voted_by"].fillna("")
    return ranked.sort_values(
        ["n_indices_agreeing", "silhouette", "behavioural_separation"],
        ascending=[False, False, False],
    ).iloc[0]


# --------------------------------------------------------------------------
# Final fits
# --------------------------------------------------------------------------
def fit_kmeans(X: np.ndarray, k: int,
               random_state: int = cfg.RANDOM_STATE) -> KMeans:
    return KMeans(n_clusters=k, n_init=50,
                  random_state=random_state).fit(X)


def fit_hierarchical(X: np.ndarray, k: int, method: str = "ward"
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """Ward linkage on the same matrix, returning labels, linkage and CCC.

    The cophenetic correlation coefficient reports how faithfully the
    dendrogram preserves the original pairwise distances; values above ~0.7
    indicate the tree is a reasonable summary of the geometry.
    """
    Z = linkage(X, method=method)
    ccc, _ = cophenet(Z, pdist(X))
    labels = fcluster(Z, t=k, criterion="maxclust") - 1
    return labels, Z, float(ccc)


def compare_partitions(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Agreement between two labelings, invariant to label permutation."""
    return {
        "adjusted_rand": float(adjusted_rand_score(a, b)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(a, b)),
    }


def stability_check(X: np.ndarray, k: int, n_runs: int = 20,
                    sample_frac: float = 0.8,
                    random_state: int = cfg.RANDOM_STATE) -> dict[str, float]:
    """Bootstrap the solution: do the same segments reappear on subsamples?

    Each run clusters an 80% subsample and compares the labels on the shared
    rows against the full-data solution via the adjusted Rand index. A mean
    ARI near 1 means the partition is a property of the data rather than of
    the particular sample.
    """
    rng = np.random.default_rng(random_state)
    base = fit_kmeans(X, k, random_state).labels_
    scores = []
    n = len(X)
    for i in range(n_runs):
        idx = rng.choice(n, size=int(sample_frac * n), replace=False)
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=int(rng.integers(1e6))).fit(X[idx])
        scores.append(adjusted_rand_score(base[idx], km.labels_))
    return {
        "mean_ari": float(np.mean(scores)),
        "std_ari": float(np.std(scores)),
        "min_ari": float(np.min(scores)),
        "n_runs": n_runs,
    }


def pca_projection(X: np.ndarray, n_components: int = 2
                   ) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=cfg.RANDOM_STATE)
    return pca.fit_transform(X), pca


def silhouette_by_cluster(X: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    s = silhouette_samples(X, labels)
    return (pd.DataFrame({"cluster": labels, "silhouette": s})
            .groupby("cluster", observed=True)["silhouette"]
            .agg(["mean", "min", "max", "size"])
            .reset_index())


__all__ = [
    "AgglomerativeClustering", "Candidate", "compare_partitions",
    "dendrogram", "elbow_k", "fit_hierarchical", "fit_kmeans",
    "pca_projection", "run_experiment", "scan_k", "select_best",
    "silhouette_by_cluster", "stability_check",
]
