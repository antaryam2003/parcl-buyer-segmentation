"""Figure generation for the EDA (Phase 4) and the results (Phases 6-8).

Every figure follows the same rules: one measure per axis, no dual axes,
recessive grid and spines, direct labels wherever a legend would otherwise be
needed, and the four-segment palette applied by segment identity rather than
by rank so a filtered chart never repaints its survivors.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from . import config as cfg

mpl.use("Agg")

#: Display order for the derived age bands, used to restore the ordering
#: that a CSV round-trip drops.
AGE_BAND_ORDER = ["<35", "35-44", "45-54", "55-64", "65+"]


def seg_colour(name: str, fallback: int = 0) -> str:
    """Colour for a segment by identity, with a stable fallback."""
    return cfg.SEGMENT_COLOURS.get(name, cfg.PALETTE[fallback %
                                                     len(cfg.PALETTE)])


def seg_marker(name: str, fallback: int = 0) -> str:
    return cfg.SEGMENT_MARKERS.get(name, cfg.MARKERS[fallback %
                                                     len(cfg.MARKERS)])


def apply_style() -> None:
    """House style: recessive chrome, generous whitespace, no top/right spine."""
    plt.rcParams.update({
        "figure.facecolor": cfg.SURFACE,
        "axes.facecolor": cfg.SURFACE,
        "savefig.facecolor": cfg.SURFACE,
        "axes.edgecolor": cfg.GRID,
        "axes.labelcolor": cfg.INK_MUTED,
        "axes.titlecolor": cfg.INK,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": cfg.GRID,
        "grid.linewidth": 0.8,
        "xtick.color": cfg.INK_MUTED,
        "ytick.color": cfg.INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "figure.dpi": cfg.DPI,
        "savefig.dpi": cfg.DPI,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
        "patch.linewidth": 0,
    })


def _save(fig: plt.Figure, name: str) -> Path:
    path = cfg.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _usd(x, _pos=None) -> str:
    if abs(x) >= 1e9:
        return f"${x/1e9:.1f}B"
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def _bar_labels(ax, bars, fmt="{:,.0f}", pad=3, horizontal=False) -> None:
    """Direct value labels - the chart should not need a lookup table.

    ``fmt`` is either a format string or a callable taking the value.
    """
    render = fmt if callable(fmt) else fmt.format
    for b in bars:
        if horizontal:
            v = b.get_width()
            ax.annotate(render(v), (v, b.get_y() + b.get_height() / 2),
                        xytext=(pad, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=9,
                        color=cfg.INK_MUTED)
        else:
            v = b.get_height()
            ax.annotate(render(v), (b.get_x() + b.get_width() / 2, v),
                        xytext=(0, pad), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9,
                        color=cfg.INK_MUTED)


# ==========================================================================
# EDA - A. Customer demographics
# ==========================================================================
def fig_age_distribution(clients: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(clients["age"], bins=range(20, 100, 5), color=cfg.NEUTRAL,
            edgecolor=cfg.SURFACE, linewidth=1.5)
    med = clients["age"].median()
    ax.axvline(med, color=cfg.INK, linewidth=1.5, linestyle="--")
    ax.annotate(f"median {med:.0f}", (med, ax.get_ylim()[1] * 0.94),
                xytext=(6, 0), textcoords="offset points",
                fontsize=9, color=cfg.INK)
    ax.set_title("Buyer age is close to uniform between 25 and 94")
    ax.set_xlabel("Age at 31 December 2025 (years)")
    ax.set_ylabel("Clients")
    return _save(fig, "eda_01_age_distribution")


def fig_demographic_mix(clients: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    specs = [
        ("client_type", "Client type", axes[0]),
        ("gender", "Gender", axes[1]),
        ("referral_channel", "Acquisition channel", axes[2]),
    ]
    for col, title, ax in specs:
        vc = clients[col].value_counts()
        bars = ax.bar(vc.index.astype(str), vc.to_numpy(),
                      color=cfg.PALETTE[0], width=0.62)
        _bar_labels(ax, bars)
        ax.set_title(title)
        ax.set_ylabel("Clients" if ax is axes[0] else "")
        ax.set_ylim(0, vc.max() * 1.18)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Who the buyers are", x=0.005, ha="left", fontsize=14,
                 fontweight="600", color=cfg.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "eda_02_demographic_mix")


def fig_geography(clients: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cvc = clients["country"].value_counts()
    bars = axes[0].barh(cvc.index[::-1].astype(str), cvc.to_numpy()[::-1],
                        color=cfg.PALETTE[0], height=0.68)
    _bar_labels(axes[0], bars, horizontal=True)
    axes[0].set_title("Clients by country")
    axes[0].set_xlim(0, cvc.max() * 1.15)
    axes[0].grid(axis="y", visible=False)

    rvc = clients["region"].value_counts().head(12)
    bars = axes[1].barh(rvc.index[::-1].astype(str), rvc.to_numpy()[::-1],
                        color=cfg.PALETTE[2], height=0.68)
    _bar_labels(axes[1], bars, horizontal=True)
    axes[1].set_title("Twelve largest regions of 57")
    axes[1].set_xlim(0, rvc.max() * 1.15)
    axes[1].grid(axis="y", visible=False)

    fig.suptitle("The book is concentrated: 77% of clients are in the USA, "
                 "32% in California alone",
                 x=0.005, ha="left", fontsize=14, fontweight="600",
                 color=cfg.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "eda_03_geography")


# ==========================================================================
# EDA - B. Buyer intent
# ==========================================================================
def fig_buyer_intent(clients: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    specs = [
        ("acquisition_purpose", "Acquisition purpose", axes[0],
         cfg.PALETTE[0]),
        ("loan_applied", "Financing applied for", axes[1], cfg.PALETTE[1]),
        ("satisfaction_score", "Satisfaction score", axes[2], cfg.PALETTE[2]),
    ]
    for col, title, ax, colour in specs:
        vc = clients[col].value_counts().sort_index() if col == \
            "satisfaction_score" else clients[col].value_counts()
        bars = ax.bar(vc.index.astype(str), vc.to_numpy(), color=colour,
                      width=0.62)
        _bar_labels(ax, bars)
        ax.set_title(title)
        ax.set_ylabel("Clients" if ax is axes[0] else "")
        ax.set_ylim(0, vc.max() * 1.18)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Stated intent: 69% buy to live in, 37% seek financing, "
                 "satisfaction is flat across all five levels",
                 x=0.005, ha="left", fontsize=14, fontweight="600",
                 color=cfg.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "eda_04_buyer_intent")


# ==========================================================================
# EDA - C. Property behaviour
# ==========================================================================
def fig_price_distribution(properties: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                             gridspec_kw={"width_ratios": [2, 1]})
    price = properties["sale_price"].dropna()
    axes[0].hist(price, bins=45, color=cfg.NEUTRAL, edgecolor=cfg.SURFACE,
                 linewidth=1.2)
    for val, lab, style in [(price.mean(), "mean", "--"),
                            (price.median(), "median", ":")]:
        axes[0].axvline(val, color=cfg.INK, linewidth=1.5, linestyle=style)
        axes[0].annotate(f"{lab} {_usd(val)}",
                         (val, axes[0].get_ylim()[1] * 0.9),
                         xytext=(6, 0), textcoords="offset points",
                         fontsize=9, color=cfg.INK)
    axes[0].set_title("Transaction price distribution")
    axes[0].set_xlabel("Sale price")
    axes[0].set_ylabel("Listings")
    axes[0].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_usd))

    box = axes[1].boxplot(
        [properties.loc[properties["unit_category"].eq(c), "sale_price"]
         .dropna() for c in ("Apartment", "Office")],
        tick_labels=["Apartment", "Office"], patch_artist=True, widths=0.5,
        medianprops={"color": cfg.INK, "linewidth": 1.8},
        flierprops={"marker": ".", "markersize": 3,
                    "markerfacecolor": cfg.INK_MUTED,
                    "markeredgecolor": "none", "alpha": 0.4},
    )
    for patch, colour in zip(box["boxes"], (cfg.PALETTE[0], cfg.PALETTE[1])):
        patch.set_facecolor(colour)
        patch.set_alpha(0.85)
    axes[1].set_title("Price by unit category")
    axes[1].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_usd))
    axes[1].grid(axis="x", visible=False)

    fig.suptitle("Prices span $97K to $737K around a $344K mean, with no "
                 "premium tail by unit type",
                 x=0.005, ha="left", fontsize=14, fontweight="600",
                 color=cfg.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, "eda_05_price_distribution")


def fig_inventory(properties: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, col, title, colour in [
        (axes[0], "unit_category", "Unit category", cfg.PALETTE[0]),
        (axes[1], "listing_status", "Listing status", cfg.PALETTE[1]),
    ]:
        vc = properties[col].value_counts()
        bars = ax.bar(vc.index.astype(str), vc.to_numpy(), color=colour,
                      width=0.55)
        _bar_labels(ax, bars, fmt="{:,.0f}")
        ax.set_title(title)
        ax.set_ylim(0, vc.max() * 1.18)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Listings")

    monthly = (properties.loc[properties["listing_status"].eq("Sold")]
               .groupby(properties["transaction_date"].dt.to_period("M"))
               .size())
    axes[2].plot(monthly.index.to_timestamp(), monthly.to_numpy(),
                 color=cfg.PALETTE[2], marker="o", markersize=4)
    axes[2].set_title("Sales per month")
    axes[2].set_ylim(0, monthly.max() * 1.15)
    axes[2].tick_params(axis="x", rotation=45)

    fig.suptitle("Inventory and sales cadence: volume steps down about a "
                 "third between 2024 and 2025",
                 x=0.005, ha="left", fontsize=14, fontweight="600",
                 color=cfg.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, "eda_06_inventory")


def fig_portfolio_size(features: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    vc = features["total_properties"].value_counts().sort_index()
    bars = axes[0].bar(vc.index.astype(str), vc.to_numpy(),
                       color=cfg.PALETTE[0], width=0.62)
    _bar_labels(axes[0], bars)
    axes[0].set_title("Units purchased per client")
    axes[0].set_xlabel("Units")
    axes[0].set_ylabel("Clients")
    axes[0].set_ylim(0, vc.max() * 1.16)
    axes[0].grid(axis="x", visible=False)

    axes[1].hist(features["total_investment"], bins=45, color=cfg.NEUTRAL,
                 edgecolor=cfg.SURFACE, linewidth=1.2)
    med = features["total_investment"].median()
    axes[1].axvline(med, color=cfg.INK, linewidth=1.5, linestyle="--")
    axes[1].annotate(f"median {_usd(med)}",
                     (med, axes[1].get_ylim()[1] * 0.92),
                     xytext=(6, 0), textcoords="offset points", fontsize=9,
                     color=cfg.INK)
    axes[1].set_title("Total capital committed per client")
    axes[1].set_xlabel("Total investment")
    axes[1].set_ylabel("Clients")
    axes[1].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_usd))

    fig.suptitle("Almost every client buys three or four units; the long "
                 "tail beyond six is where the capital concentrates",
                 x=0.005, ha="left", fontsize=14, fontweight="600",
                 color=cfg.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, "eda_07_portfolio_size")


# ==========================================================================
# EDA - D. Buyer-investment relationships
# ==========================================================================
def fig_intent_vs_investment(features: pd.DataFrame) -> Path:
    """The null result: stated attributes do not predict spend."""
    pairs = [("client_type", "Client type"),
             ("acquisition_purpose", "Purpose"),
             ("loan_applied", "Financing"),
             ("age_band", "Age band")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)
    overall = features["total_investment"].mean()
    features = features.copy()
    # age_band is an ordered category in memory but degrades to plain text
    # on a CSV round-trip, which would sort "<35" to the end. Restore it.
    features["age_band"] = pd.Categorical(
        features["age_band"], categories=AGE_BAND_ORDER, ordered=True)
    for ax, (col, title) in zip(axes, pairs):
        m = features.groupby(col, observed=True)["total_investment"].mean()
        bars = ax.bar(m.index.astype(str), m.to_numpy(),
                      color=cfg.PALETTE[0], width=0.6)
        _bar_labels(ax, bars, fmt=_usd)
        ax.axhline(overall, color=cfg.INK, linewidth=1.4, linestyle="--")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_usd))
    # Headroom so the direct labels clear the reference line, and the line's
    # own caption sits above every mark rather than colliding with one.
    axes[0].set_ylim(0, overall * 1.42)
    axes[0].set_ylabel("Mean total investment")
    axes[3].annotate(f"overall mean {_usd(overall)}", (0.97, 0.97),
                     xycoords="axes fraction", ha="right", va="top",
                     fontsize=9, color=cfg.INK)
    fig.suptitle("Declared buyer attributes are flat against spend - every "
                 "bar sits on the overall mean",
                 x=0.005, ha="left", fontsize=14, fontweight="600",
                 color=cfg.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save(fig, "eda_08_intent_vs_investment")


def fig_correlation_heatmap(features: pd.DataFrame,
                            columns: list[str]) -> Path:
    corr = features[columns].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.55 else cfg.INK)
    ax.set_title("Behaviour correlates with behaviour; demographics correlate "
                 "with nothing")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    fig.tight_layout()
    return _save(fig, "eda_09_correlation")


# ==========================================================================
# Results - model selection
# ==========================================================================
def fig_elbow_silhouette(scan: pd.DataFrame, chosen_k: int) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    panels = [
        ("inertia", "Elbow: within-cluster sum of squares", cfg.PALETTE[0],
         False),
        ("silhouette", "Silhouette score (higher is better)",
         cfg.PALETTE[1], True),
        ("davies_bouldin", "Davies-Bouldin index (lower is better)",
         cfg.PALETTE[2], True),
    ]
    for ax, (col, title, colour, mark) in zip(axes, panels):
        ax.plot(scan["k"], scan[col], color=colour, marker="o", markersize=5)
        if mark:
            best = (scan.loc[scan[col].idxmax()] if col == "silhouette"
                    else scan.loc[scan[col].idxmin()])
            ax.scatter([best["k"]], [best[col]], s=140, facecolor="none",
                       edgecolor=cfg.INK, linewidth=1.6, zorder=5)
        ax.axvline(chosen_k, color=cfg.INK_MUTED, linewidth=1,
                   linestyle=":", zorder=0)
        ax.set_title(title)
        ax.set_xlabel("Number of clusters (K)")
        ax.set_xticks(list(scan["k"]))
    axes[0].set_ylabel("Inertia")
    axes[0].annotate(f"K = {chosen_k} selected", (chosen_k, scan["inertia"]
                     .max()), xytext=(8, -6), textcoords="offset points",
                     fontsize=9, color=cfg.INK)
    fig.suptitle(f"All three internal indices select K = {chosen_k}",
                 x=0.005, ha="left", fontsize=14, fontweight="600",
                 color=cfg.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _save(fig, "model_01_elbow_silhouette")


def fig_pca_scatter(coords: np.ndarray, labels: np.ndarray,
                    names: dict[int, str], pca) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 7))
    order = sorted(names, key=lambda c: cfg.SEGMENT_ORDER.index(names[c])
                   if names[c] in cfg.SEGMENT_ORDER else 99)
    for i, cl in enumerate(order):
        m = labels == cl
        ax.scatter(coords[m, 0], coords[m, 1], s=26,
                   color=seg_colour(names[cl], i),
                   marker=seg_marker(names[cl], i),
                   alpha=0.75, linewidths=0.6, edgecolors=cfg.SURFACE,
                   label=f"{names[cl]} (n={int(m.sum())})")
        cx, cy = coords[m, 0].mean(), coords[m, 1].mean()
        ax.annotate(names[cl], (cx, cy), fontsize=9.5, fontweight="600",
                    color=cfg.INK, ha="center",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor":
                          cfg.SURFACE, "edgecolor": cfg.GRID, "alpha": 0.9})
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 - portfolio scale ({ev[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 - unit size and ticket ({ev[1]:.0%} of variance)")
    ax.set_title("Segments occupy distinct regions of the behaviour space")
    ax.legend(loc="upper left", ncols=1)
    fig.tight_layout()
    return _save(fig, "model_02_pca_scatter")


def fig_dendrogram(Z: np.ndarray, chosen_k: int, ccc: float) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax,
               color_threshold=Z[-(chosen_k - 1), 2],
               above_threshold_color=cfg.INK_MUTED, no_labels=True)
    ax.axhline(Z[-(chosen_k - 1), 2], color=cfg.INK, linewidth=1.4,
               linestyle="--")
    ax.annotate(f"cut for K = {chosen_k}", (0.01, Z[-(chosen_k - 1), 2]),
                xycoords=("axes fraction", "data"), xytext=(0, 6),
                textcoords="offset points", fontsize=9, color=cfg.INK)
    ax.set_title(f"Ward dendrogram, last 30 merges "
                 f"(cophenetic correlation {ccc:.2f})")
    ax.set_ylabel("Merge distance")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, "model_03_dendrogram")


def fig_segment_sizes(profile: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    prof = profile.sort_values("n_clients", ascending=True)
    colours = [seg_colour(s, i) for i, s in enumerate(prof["segment"])]

    bars = axes[0].barh(prof["segment"], prof["n_clients"], color=colours,
                        height=0.66)
    _bar_labels(axes[0], bars, fmt="{:,.0f}", horizontal=True)
    axes[0].set_title("Clients per segment")
    axes[0].set_xlim(0, prof["n_clients"].max() * 1.16)
    axes[0].grid(axis="y", visible=False)

    bars = axes[1].barh(prof["segment"], prof["capital_committed"],
                        color=colours, height=0.66)
    for b, v in zip(bars, prof["capital_committed"]):
        axes[1].annotate(_usd(v), (v, b.get_y() + b.get_height() / 2),
                         xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=9, color=cfg.INK_MUTED)
    axes[1].set_title("Capital committed per segment")
    axes[1].set_xlim(0, prof["capital_committed"].max() * 1.2)
    axes[1].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_usd))
    axes[1].set_yticklabels([])
    axes[1].grid(axis="y", visible=False)

    fig.suptitle("Segment scale: headcount against capital",
                 x=0.005, ha="left", fontsize=14, fontweight="600",
                 color=cfg.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, "model_04_segment_sizes")


def fig_segment_fingerprint(profile: pd.DataFrame,
                            features: pd.DataFrame,
                            columns: list[str]) -> Path:
    """Segment means in overall standard deviations - the interpretation key."""
    mu, sd = features[columns].mean(), features[columns].std()
    z = (profile[columns] - mu) / sd

    # The accumulator tier sits nearly 5 SD out on portfolio scale, which
    # would flatten the other three segments to invisibility on a shared
    # axis. The lower panel repeats the chart clipped to +/-1 SD so the
    # mainstream segments are readable without hiding the outlier.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.35, 1]})
    x = np.arange(len(columns))
    width = 0.8 / len(z)
    for ax in axes:
        for i, (cl, row) in enumerate(z.iterrows()):
            ax.bar(x + i * width - 0.4 + width / 2, row.to_numpy(),
                   width * 0.88,
                   color=seg_colour(profile.loc[cl, "segment"], i),
                   label=profile.loc[cl, "segment"])
        ax.axhline(0, color=cfg.INK, linewidth=1.2)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("SD from overall mean")

    axes[0].set_title("Segment fingerprints: what actually separates the "
                      "groups")
    axes[1].set_ylim(-1.15, 1.15)
    axes[1].set_title("Same chart, clipped to +/-1 SD: the three mainstream "
                      "segments", fontsize=11)
    axes[1].set_xticks(x, [c.replace("_", " ") for c in columns], rotation=30,
                       ha="right")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:len(z)], labels[:len(z)], ncols=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, "model_05_segment_fingerprint")


def fig_segment_geography(segmented: pd.DataFrame) -> Path:
    top = segmented["region"].value_counts().head(10).index
    sub = segmented[segmented["region"].isin(top)]
    ct = (pd.crosstab(sub["region"], sub["segment"], normalize="index")
          .loc[top])
    ct = ct[[s for s in cfg.SEGMENT_ORDER if s in ct.columns]
            + [s for s in ct.columns if s not in cfg.SEGMENT_ORDER]]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    left = np.zeros(len(ct))
    for i, seg in enumerate(ct.columns):
        vals = ct[seg].to_numpy()
        ax.barh(ct.index, vals, left=left, height=0.66,
                color=seg_colour(seg, i), label=seg,
                edgecolor=cfg.SURFACE, linewidth=2)
        left += vals
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_title("Segment mix is essentially identical in every major region")
    ax.set_xlabel("Share of clients in region")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, "model_06_segment_geography")
