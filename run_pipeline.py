"""End-to-end pipeline: raw CSVs -> segmented clients, figures and tables.

Run with ``python run_pipeline.py``. Every artefact the research paper and
the Streamlit dashboard consume is produced here, so the whole analysis is
reproducible from a clean checkout with one command.
"""
from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from src import config as cfg
from src import visuals as vz
from src.clustering import (behavioural_separation, compare_partitions,
                            elbow_k, eta_squared, fit_hierarchical,
                            fit_kmeans, flag_redundancy, pca_projection,
                            run_experiment, scan_k, select_best,
                            silhouette_by_cluster, stability_check)
from src.data_cleaning import load_and_clean
from src.feature_engineering import (build_client_features,
                                     feature_dictionary)
from src.interpretation import (attach_segments, discriminating_features,
                                relative_profile, save_outputs)
from src.preprocessing import FEATURE_SETS, build_matrix, label_encode

warnings.filterwarnings("ignore", category=FutureWarning)

#: Features shown on the segment fingerprint chart, in reporting order.
FINGERPRINT = [
    "total_properties", "total_investment", "avg_property_price",
    "avg_floor_area", "total_area", "unique_towers", "active_months",
    "purchase_span_days", "price_dispersion", "office_share",
    "age", "satisfaction_score",
]

CORR_COLUMNS = [
    "age", "satisfaction_score", "total_properties", "total_investment",
    "avg_property_price", "price_dispersion", "avg_floor_area", "total_area",
    "avg_price_per_sqft", "unique_towers", "active_months",
    "purchase_span_days", "office_share",
]


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def main() -> None:
    vz.apply_style()
    summary: dict[str, object] = {}

    # ---- Phase 1: cleaning ------------------------------------------
    banner("PHASE 1  Data cleaning")
    clients, properties, report = load_and_clean()
    print(report.summary())
    summary["cleaning"] = {k: (str(v) if not isinstance(v, (int, float, bool))
                               else v)
                           for k, v in report.__dict__.items()
                           if k != "notes"}
    summary["cleaning_notes"] = report.notes

    # ---- Phases 2-3: feature engineering -----------------------------
    banner("PHASE 2-3  Merge and client-level feature engineering")
    features = build_client_features(clients, properties)
    features, _encoders = label_encode(features)
    features.to_csv(cfg.CLIENT_FEATURES, index=False)
    feature_dictionary().to_csv(cfg.TABLES_DIR / "feature_dictionary.csv",
                                index=False)
    print(f"client feature table: {features.shape[0]} rows x "
          f"{features.shape[1]} columns")
    print(f"units aggregated: {int(features['total_properties'].sum()):,} "
          f"(expected 7,305)")
    print(f"capital aggregated: ${features['total_investment'].sum():,.2f}")
    summary["n_clients"] = int(len(features))
    summary["n_properties"] = int(len(properties))
    summary["units_sold"] = int(features["total_properties"].sum())
    summary["total_capital"] = float(features["total_investment"].sum())

    # ---- Phase 4: EDA -------------------------------------------------
    banner("PHASE 4  Exploratory data analysis")
    figs = [
        vz.fig_age_distribution(features),
        vz.fig_demographic_mix(features),
        vz.fig_geography(features),
        vz.fig_buyer_intent(features),
        vz.fig_price_distribution(properties),
        vz.fig_inventory(properties),
        vz.fig_portfolio_size(features),
        vz.fig_intent_vs_investment(features),
        vz.fig_correlation_heatmap(features, CORR_COLUMNS),
    ]
    for f in figs:
        print("  wrote", f.name)

    features[CORR_COLUMNS].corr().round(3).to_csv(
        cfg.TABLES_DIR / "correlation_matrix.csv")
    features[CORR_COLUMNS].describe().T.round(3).to_csv(
        cfg.TABLES_DIR / "numeric_summary.csv")

    # ---- Phase 5-6: model selection -----------------------------------
    banner("PHASE 5-6  Encoding, scaling and cluster-count selection")
    experiment = run_experiment(features)
    experiment.to_csv(cfg.TABLES_DIR / "model_selection_experiment.csv",
                      index=False)
    best = select_best(experiment)
    fs_name, scaler, k = best["feature_set"], best["scaler"], int(best["k"])
    print(f"grid searched {len(experiment)} configurations "
          f"({experiment['feature_set'].nunique()} feature sets x "
          f"{experiment['scaler'].nunique()} scalers x "
          f"{experiment['k'].nunique()} values of K)")

    # The unconstrained silhouette winner is reported alongside the chosen
    # configuration so the rejection is visible rather than implicit.
    raw_winner = experiment.sort_values("silhouette",
                                        ascending=False).iloc[0]
    print(f"\nhighest silhouette overall : {raw_winner['feature_set']} / "
          f"{raw_winner['scaler']} / K={int(raw_winner['k'])} "
          f"sil={raw_winner['silhouette']:.4f} "
          f"-> flag_redundancy={raw_winner['flag_redundancy']:.3f}, "
          f"behavioural_separation={raw_winner['behavioural_separation']:.4f}")
    if not raw_winner["substantive"]:
        print("  REJECTED: reproduces the binary-flag crosstab rather than "
              "discovering behaviour.")

    # The chosen configuration is the one the guarded search returns; the
    # printout states it explicitly rather than assuming a hard-coded pick.
    print(f"\nselected configuration     : {fs_name} / {scaler} / K={k}")
    print(f"  silhouette {best['silhouette']:.4f} | "
          f"davies-bouldin {best['davies_bouldin']:.4f} | "
          f"calinski-harabasz {best['calinski_harabasz']:.1f}")
    print(f"  behavioural separation {best['behavioural_separation']:.3f} | "
          f"flag redundancy {best['flag_redundancy']:.3f}")

    X, feature_names, preprocessor = build_matrix(
        features, FEATURE_SETS[fs_name], scaler)
    scan = scan_k(X)
    scan.to_csv(cfg.TABLES_DIR / "k_scan.csv", index=False)
    ek = elbow_k(scan["k"].tolist(), scan["inertia"].tolist())
    print(f"\n  elbow (max distance to chord)  K = {ek}")
    print(f"  silhouette maximum             K = "
          f"{int(scan.loc[scan['silhouette'].idxmax(), 'k'])}")
    print(f"  davies-bouldin minimum         K = "
          f"{int(scan.loc[scan['davies_bouldin'].idxmin(), 'k'])}")
    print(f"  calinski-harabasz maximum      K = "
          f"{int(scan.loc[scan['calinski_harabasz'].idxmax(), 'k'])}")
    print(f"  indices agreeing on K={k}: "
          f"{best.get('voted_by', 'n/a')} "
          f"({int(best.get('n_indices_agreeing', 0))} of 3)")
    print("  wrote", vz.fig_elbow_silhouette(scan, k).name)

    # ---- Phase 6: final K-Means ---------------------------------------
    banner("PHASE 6  Final K-Means fit")
    km = fit_kmeans(X, k)
    labels = km.labels_
    print("cluster sizes:", np.bincount(labels).tolist())
    print("\nper-cluster silhouette:")
    print(silhouette_by_cluster(X, labels).round(3).to_string(index=False))

    stability = stability_check(X, k, n_runs=100)
    print(f"\nbootstrap stability over 100 x 80% subsamples: "
          f"mean ARI {stability['mean_ari']:.3f} "
          f"(sd {stability['std_ari']:.3f}, min {stability['min_ari']:.3f})")

    # ---- Phase 7: hierarchical validation -----------------------------
    banner("PHASE 7  Hierarchical validation")
    hier_labels, Z, ccc = fit_hierarchical(X, k)
    agreement = compare_partitions(labels, hier_labels)
    print(f"ward linkage, cophenetic correlation {ccc:.3f}")
    print(f"hierarchical cluster sizes: {np.bincount(hier_labels).tolist()}")
    print(f"K-Means vs hierarchical: ARI "
          f"{agreement['adjusted_rand']:.3f}, AMI "
          f"{agreement['adjusted_mutual_info']:.3f}")
    print("  wrote", vz.fig_dendrogram(Z, k, ccc).name)

    # ---- Phase 8: interpretation --------------------------------------
    banner("PHASE 8  Cluster interpretation and naming")
    segmented, profile, names = attach_segments(features, labels)
    segmented["hierarchical_cluster"] = hier_labels
    rel = relative_profile(profile, features)
    disc = discriminating_features(profile, features, top_n=8)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    show = ["segment", "n_clients", "pct_clients", "pct_capital",
            "total_properties", "total_investment", "avg_property_price",
            "avg_floor_area", "purchase_span_days", "office_share",
            "age", "satisfaction_score", "loan_flag", "is_investment"]
    print(profile[show].round(3).to_string())
    print("\nmost discriminating features (range of segment means, in SD):")
    print(disc.round(2).to_string(index=False))

    rel.to_csv(cfg.TABLES_DIR / "relative_profile.csv")
    disc.to_csv(cfg.TABLES_DIR / "discriminating_features.csv", index=False)
    eta_squared(features, labels, FINGERPRINT).round(4).to_csv(
        cfg.TABLES_DIR / "eta_squared.csv", header=["eta_squared"])

    coords, pca = pca_projection(X, 2)
    for f in (vz.fig_pca_scatter(coords, labels, names, pca),
              vz.fig_segment_sizes(profile),
              vz.fig_segment_fingerprint(profile, features, FINGERPRINT),
              vz.fig_segment_geography(segmented)):
        print("  wrote", f.name)

    segmented["pc1"] = coords[:, 0]
    segmented["pc2"] = coords[:, 1]
    save_outputs(segmented, profile)

    # ---- Persist the model ---------------------------------------------
    banner("Persisting model artefacts")
    joblib.dump(preprocessor, cfg.MODELS_DIR / "preprocessor.pkl")
    joblib.dump(km, cfg.MODELS_DIR / "kmeans.pkl")
    joblib.dump(pca, cfg.MODELS_DIR / "pca.pkl")
    for p in ("preprocessor.pkl", "kmeans.pkl", "pca.pkl"):
        print("  wrote models/" + p)

    summary["model"] = {
        "feature_set": fs_name,
        "scaler": scaler,
        "k": k,
        "n_model_features": int(X.shape[1]),
        "model_features": feature_names,
        "silhouette": float(best["silhouette"]),
        "davies_bouldin": float(best["davies_bouldin"]),
        "calinski_harabasz": float(best["calinski_harabasz"]),
        "elbow_k": int(ek),
        "behavioural_separation": float(behavioural_separation(features,
                                                               labels)),
        "flag_redundancy": float(flag_redundancy(features, labels)),
        "pca_explained": [float(v) for v in pca.explained_variance_ratio_],
        "stability": stability,
        "hierarchical": {"cophenetic_correlation": ccc, **agreement},
        "segments": {str(i): n for i, n in names.items()},
        "rejected_winner": {
            "feature_set": str(raw_winner["feature_set"]),
            "scaler": str(raw_winner["scaler"]),
            "k": int(raw_winner["k"]),
            "silhouette": float(raw_winner["silhouette"]),
            "flag_redundancy": float(raw_winner["flag_redundancy"]),
            "behavioural_separation":
                float(raw_winner["behavioural_separation"]),
        },
    }
    (cfg.OUTPUTS_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    banner("DONE")
    print(f"segmented clients -> {cfg.SEGMENTED_CLIENTS}")
    print(f"cluster profiles  -> {cfg.CLUSTER_PROFILES}")
    print(f"figures           -> {cfg.FIGURES_DIR} "
          f"({len(list(cfg.FIGURES_DIR.glob('*.png')))} files)")
    print(f"tables            -> {cfg.TABLES_DIR} "
          f"({len(list(cfg.TABLES_DIR.glob('*.csv')))} files)")


if __name__ == "__main__":
    main()
