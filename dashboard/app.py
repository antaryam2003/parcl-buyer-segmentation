"""Parcl Buyer Intelligence - Streamlit dashboard.

Implements the four modules the PRD specifies (segmentation overview,
investor behaviour, geographic analysis, segment insights) over the
segmented client table produced by ``run_pipeline.py``.

The app reads the pipeline's outputs rather than re-fitting the model, so
what is displayed is exactly what the research paper reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.interpretation import SEGMENT_LIBRARY  # noqa: E402

# --------------------------------------------------------------------------
# Palette - imported from src.config so the dashboard and the research
# paper's figures are guaranteed to use the same colours for the same
# segments. The set is validated all-pairs for colour-vision separation
# against a light surface; the app pins a light theme in
# .streamlit/config.toml so those guarantees hold.
# --------------------------------------------------------------------------
PALETTE = cfg.PALETTE
SURFACE = cfg.SURFACE
INK = cfg.INK
INK_MUTED = cfg.INK_MUTED
GRID = cfg.GRID
SEGMENT_ORDER = cfg.SEGMENT_ORDER
SEGMENT_COLOURS = cfg.SEGMENT_COLOURS

st.set_page_config(page_title="Parcl Buyer Intelligence",
                   page_icon="🏙️", layout="wide")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    seg_path = ROOT / "outputs" / "segmented_clients.csv"
    prof_path = ROOT / "outputs" / "cluster_profiles.csv"
    if not seg_path.exists():
        st.error(
            "outputs/segmented_clients.csv is missing. Run "
            "`python run_pipeline.py` first to build the model outputs."
        )
        st.stop()
    seg = pd.read_csv(seg_path)
    prof = pd.read_csv(prof_path, index_col=0)
    return seg, prof


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": INK_MUTED, "size": 12},
        title={"font": {"color": INK, "size": 15}, "x": 0, "xanchor": "left"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.28,
                "x": 0, "title": ""},
        hoverlabel={"bgcolor": SURFACE, "font_size": 12},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def usd(x: float) -> str:
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


segmented, profile = load_data()

# --------------------------------------------------------------------------
# Header and filters
# --------------------------------------------------------------------------
st.title("Parcl Buyer Intelligence")
st.caption(
    "Machine-learning buyer segmentation and investment profiling. "
    "2,000 clients matched to 7,305 sold transactions, segmented with "
    "K-Means (K=4) on ten client-level investment-behaviour features."
)

with st.sidebar:
    st.header("Filters")
    st.caption("Every chart and metric below responds to these.")

    def multi(label: str, column: str) -> list:
        options = sorted(segmented[column].dropna().unique().tolist())
        return st.multiselect(label, options, default=options)

    f_country = multi("Country", "country")
    f_purpose = multi("Acquisition purpose", "acquisition_purpose")
    f_type = multi("Client type", "client_type")
    f_segment = st.multiselect(
        "Segment",
        [s for s in SEGMENT_ORDER if s in set(segmented["segment"])],
        default=[s for s in SEGMENT_ORDER if s in set(segmented["segment"])],
    )
    f_loan = multi("Financing applied", "loan_applied")
    f_referral = multi("Referral channel", "referral_channel")

    regions_available = sorted(
        segmented.loc[segmented["country"].isin(f_country), "region"]
        .dropna().unique().tolist()
    )
    f_region = st.multiselect("Region", regions_available,
                              default=regions_available)

    st.divider()
    if st.button("Reset filters", width="stretch"):
        st.cache_data.clear()
        st.rerun()

mask = (
    segmented["country"].isin(f_country)
    & segmented["region"].isin(f_region)
    & segmented["acquisition_purpose"].isin(f_purpose)
    & segmented["client_type"].isin(f_type)
    & segmented["segment"].isin(f_segment)
    & segmented["loan_applied"].isin(f_loan)
    & segmented["referral_channel"].isin(f_referral)
)
view = segmented[mask]

if view.empty:
    st.warning("No clients match the current filters. Widen the selection "
               "in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------
total_all = len(segmented)
cap_all = segmented["total_investment"].sum()
cols = st.columns(5)
cols[0].metric("Buyers", f"{len(view):,}",
               f"{len(view) / total_all:.0%} of book")
cols[1].metric("Capital committed", usd(view["total_investment"].sum()),
               f"{view['total_investment'].sum() / cap_all:.0%} of total")
cols[2].metric("Avg per buyer", usd(view["total_investment"].mean()))
cols[3].metric("Units purchased", f"{int(view['total_properties'].sum()):,}")
cols[4].metric("Avg ticket", usd(view["avg_property_price"].mean()))

st.divider()

tabs = st.tabs([
    "Segmentation overview",
    "Investor behaviour",
    "Geographic analysis",
    "Segment insights",
    "Client explorer",
])

# ==========================================================================
# 1. Buyer segmentation overview
# ==========================================================================
with tabs[0]:
    st.subheader("Buyer segmentation overview")
    left, right = st.columns([1, 1])

    counts = (view["segment"].value_counts()
              .reindex(SEGMENT_ORDER).dropna().reset_index())
    counts.columns = ["segment", "clients"]
    fig = px.bar(counts, x="clients", y="segment", orientation="h",
                 color="segment", color_discrete_map=SEGMENT_COLOURS,
                 text="clients", title="Clients per segment")
    fig.update_traces(textposition="outside",
                      hovertemplate="%{y}<br>%{x:,} clients<extra></extra>")
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Clients")
    left.plotly_chart(style_fig(fig), width="stretch")

    cap = (view.groupby("segment", observed=True)["total_investment"].sum()
           .reindex(SEGMENT_ORDER).dropna().reset_index())
    cap.columns = ["segment", "capital"]
    cap["label"] = cap["capital"].map(usd)
    fig = px.bar(cap, x="capital", y="segment", orientation="h",
                 color="segment", color_discrete_map=SEGMENT_COLOURS,
                 text="label", title="Capital committed per segment")
    fig.update_traces(textposition="outside",
                      hovertemplate="%{y}<br>%{x:$,.0f}<extra></extra>")
    fig.update_layout(showlegend=False, yaxis_title="",
                      xaxis_title="Total investment (USD)")
    right.plotly_chart(style_fig(fig), width="stretch")

    st.markdown("**Share of clients against share of capital**")
    share = (view.groupby("segment", observed=True)
             .agg(clients=("client_id", "size"),
                  capital=("total_investment", "sum"))
             .reindex(SEGMENT_ORDER).dropna())
    share["client_share"] = share["clients"] / share["clients"].sum()
    share["capital_share"] = share["capital"] / share["capital"].sum()
    share["over_index"] = share["capital_share"] / share["client_share"]
    st.dataframe(
        share.assign(
            clients=lambda d: d["clients"].map("{:,.0f}".format),
            capital=lambda d: d["capital"].map(usd),
            client_share=lambda d: d["client_share"].map("{:.1%}".format),
            capital_share=lambda d: d["capital_share"].map("{:.1%}".format),
            over_index=lambda d: d["over_index"].map("{:.2f}x".format),
        ).rename(columns={
            "clients": "Clients", "capital": "Capital",
            "client_share": "Share of clients",
            "capital_share": "Share of capital",
            "over_index": "Capital over-index"}),
        width="stretch",
    )
    st.caption(
        "Over-index above 1.00x means the segment holds more capital than "
        "its headcount implies. This table is the accessible equivalent of "
        "the two charts above."
    )

# ==========================================================================
# 2. Investor behaviour
# ==========================================================================
with tabs[1]:
    st.subheader("Investor behaviour by segment")
    c1, c2 = st.columns(2)

    fig = px.box(view, x="segment", y="total_investment", color="segment",
                 color_discrete_map=SEGMENT_COLOURS,
                 category_orders={"segment": SEGMENT_ORDER},
                 points=False, title="Capital committed per client")
    fig.update_layout(showlegend=False, xaxis_title="",
                      yaxis_title="Total investment (USD)")
    fig.update_xaxes(tickangle=-15)
    c1.plotly_chart(style_fig(fig, 400), width="stretch")

    fig = px.box(view, x="segment", y="avg_property_price", color="segment",
                 color_discrete_map=SEGMENT_COLOURS,
                 category_orders={"segment": SEGMENT_ORDER},
                 points=False, title="Average ticket size per client")
    fig.update_layout(showlegend=False, xaxis_title="",
                      yaxis_title="Average property price (USD)")
    fig.update_xaxes(tickangle=-15)
    c2.plotly_chart(style_fig(fig, 400), width="stretch")

    c3, c4 = st.columns(2)
    units = (view.groupby("segment", observed=True)["total_properties"]
             .mean().reindex(SEGMENT_ORDER).dropna().reset_index())
    fig = px.bar(units, x="segment", y="total_properties", color="segment",
                 color_discrete_map=SEGMENT_COLOURS,
                 text=units["total_properties"].round(2),
                 title="Mean units purchased")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="",
                      yaxis_title="Units per client")
    fig.update_xaxes(tickangle=-15)
    c3.plotly_chart(style_fig(fig, 380), width="stretch")

    mix = (view.groupby("segment", observed=True)[
        ["apartment_share", "office_share"]].mean()
        .reindex(SEGMENT_ORDER).dropna().reset_index()
        .melt(id_vars="segment", var_name="asset", value_name="share"))
    mix["asset"] = mix["asset"].map({"apartment_share": "Apartment",
                                     "office_share": "Office"})
    fig = px.bar(mix, x="segment", y="share", color="asset", barmode="stack",
                 color_discrete_sequence=[PALETTE[0], PALETTE[1]],
                 title="Residential / commercial unit mix")
    fig.update_layout(xaxis_title="", yaxis_title="Share of units",
                      yaxis_tickformat=".0%")
    fig.update_traces(marker_line_color=SURFACE, marker_line_width=2)
    fig.update_xaxes(tickangle=-15)
    c4.plotly_chart(style_fig(fig, 380), width="stretch")

    st.markdown("**Financing and stated intent by segment**")
    rates = (view.groupby("segment", observed=True)
             .agg(**{"Loan applied": ("loan_flag", "mean"),
                     "Investment purpose": ("is_investment", "mean"),
                     "Corporate client": ("is_company", "mean")})
             .reindex(SEGMENT_ORDER).dropna())
    st.dataframe(rates.map("{:.1%}".format), width="stretch")
    st.info(
        "These three rates are close to flat across segments. That is a "
        "finding, not a rendering fault: in this dataset the attributes a "
        "buyer declares carry almost no information about what they go on "
        "to purchase, so behaviour has to be measured from transactions.",
        icon=":material/info:",
    )

# ==========================================================================
# 3. Geographic analysis
# ==========================================================================
with tabs[2]:
    st.subheader("Geographic buyer analysis")

    by_country = (view.groupby("country", observed=True)
                  .agg(clients=("client_id", "size"),
                       capital=("total_investment", "sum"),
                       avg_investment=("total_investment", "mean"))
                  .reset_index().sort_values("clients", ascending=False))

    iso = {"USA": "USA", "UK": "GBR", "Canada": "CAN", "Germany": "DEU",
           "France": "FRA", "Belgium": "BEL", "Mexico": "MEX",
           "Australia": "AUS", "Russia": "RUS", "Denmark": "DNK"}
    by_country["iso3"] = by_country["country"].map(iso)

    fig = px.choropleth(
        by_country.dropna(subset=["iso3"]), locations="iso3",
        color="clients", hover_name="country",
        hover_data={"iso3": False, "clients": ":,",
                    "capital": ":$,.0f"},
        color_continuous_scale=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                "#256abf", "#184f95", "#0d366b"],
        title="Clients by country",
    )
    # Draw the rest of the world in a recessive neutral so the shaded
    # countries read as "these have clients" rather than as the only land.
    fig.update_geos(showframe=False, showcoastlines=False,
                    projection_type="natural earth",
                    bgcolor=SURFACE, showland=True, landcolor="#f0efec",
                    showcountries=True, countrycolor=SURFACE,
                    showocean=False, subunitcolor=SURFACE)
    fig.update_layout(coloraxis_colorbar={"title": "Clients"})
    st.plotly_chart(style_fig(fig, 460), width="stretch")

    g1, g2 = st.columns([1, 1])
    top_regions = (view["region"].value_counts().head(12)
                   .rename_axis("region").reset_index(name="clients"))
    fig = px.bar(top_regions.sort_values("clients"), x="clients", y="region",
                 orientation="h", text="clients",
                 color_discrete_sequence=[PALETTE[0]],
                 title="Twelve largest regions")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Clients")
    g1.plotly_chart(style_fig(fig, 440), width="stretch")

    sub = view[view["region"].isin(top_regions["region"])]
    ct = (pd.crosstab(sub["region"], sub["segment"], normalize="index")
          .reindex(columns=[s for s in SEGMENT_ORDER
                            if s in sub["segment"].unique()])
          .loc[top_regions.sort_values("clients")["region"]])
    fig = go.Figure()
    for seg in ct.columns:
        fig.add_bar(y=ct.index, x=ct[seg], name=seg, orientation="h",
                    marker_color=SEGMENT_COLOURS[seg],
                    marker_line_color=SURFACE, marker_line_width=2,
                    hovertemplate="%{y}<br>" + seg +
                    ": %{x:.1%}<extra></extra>")
    fig.update_layout(barmode="stack", title="Segment mix within each region",
                      xaxis_tickformat=".0%", xaxis_title="Share of clients",
                      yaxis_title="")
    st.plotly_chart(style_fig(fig, 440), width="stretch")
    g2.caption("")

    st.dataframe(
        by_country.assign(
            clients=lambda d: d["clients"].map("{:,.0f}".format),
            capital=lambda d: d["capital"].map(usd),
            avg_investment=lambda d: d["avg_investment"].map(usd),
        ).drop(columns="iso3").rename(columns={
            "country": "Country", "clients": "Clients",
            "capital": "Capital committed",
            "avg_investment": "Average per buyer"}),
        width="stretch", hide_index=True,
    )

# ==========================================================================
# 4. Segment insights
# ==========================================================================
with tabs[3]:
    st.subheader("Segment insights")
    present = [s for s in SEGMENT_ORDER if s in set(view["segment"])]
    choice = st.selectbox("Segment", present)
    sub = view[view["segment"] == choice]
    info = SEGMENT_LIBRARY.get(choice, {})

    m = st.columns(4)
    m[0].metric("Clients", f"{len(sub):,}")
    m[1].metric("Avg capital committed", usd(sub["total_investment"].mean()))
    m[2].metric("Avg ticket", usd(sub["avg_property_price"].mean()))
    m[3].metric("Avg units", f"{sub['total_properties'].mean():.2f}")

    m = st.columns(4)
    m[0].metric("Avg unit size", f"{sub['avg_floor_area'].mean():,.0f} sqft")
    m[1].metric("Financing rate", f"{sub['loan_flag'].mean():.0%}")
    m[2].metric("Investment-purpose rate",
                f"{sub['is_investment'].mean():.0%}")
    m[3].metric("Avg buying window",
                f"{sub['purchase_span_days'].mean():.0f} days")

    if info:
        st.markdown(f"**What defines this segment.** {info['thesis']}")
        st.markdown(f"**Primary signal.** {info['signal']}")
        st.success(f"**Recommended strategy.** {info['strategy']}",
                   icon=":material/lightbulb:")

    st.markdown("**How this segment compares to the whole book**")
    comp_cols = ["total_properties", "total_investment", "avg_property_price",
                 "avg_floor_area", "total_area", "unique_towers",
                 "purchase_span_days", "office_share", "age",
                 "satisfaction_score"]
    comp = pd.DataFrame({
        "Segment mean": sub[comp_cols].mean(),
        "Book mean": segmented[comp_cols].mean(),
    })
    comp["Index vs book"] = (comp["Segment mean"] / comp["Book mean"])
    st.dataframe(
        comp.assign(**{
            "Segment mean": comp["Segment mean"].map("{:,.2f}".format),
            "Book mean": comp["Book mean"].map("{:,.2f}".format),
            "Index vs book": comp["Index vs book"].map("{:.2f}x".format),
        }),
        width="stretch",
    )

# ==========================================================================
# 5. Client explorer
# ==========================================================================
with tabs[4]:
    st.subheader("Client explorer")
    st.caption(f"{len(view):,} clients match the current filters.")
    show = ["client_id", "full_name", "segment", "client_type", "country",
            "region", "age", "acquisition_purpose", "loan_applied",
            "referral_channel", "satisfaction_score", "total_properties",
            "total_investment", "avg_property_price", "avg_floor_area",
            "office_share", "purchase_span_days"]
    st.dataframe(
        view[show].sort_values("total_investment", ascending=False),
        width="stretch", hide_index=True, height=460,
        column_config={
            "total_investment": st.column_config.NumberColumn(
                "Total investment", format="$%.0f"),
            "avg_property_price": st.column_config.NumberColumn(
                "Avg price", format="$%.0f"),
            "avg_floor_area": st.column_config.NumberColumn(
                "Avg sqft", format="%.0f"),
            "office_share": st.column_config.NumberColumn(
                "Office share", format="%.2f"),
        },
    )
    st.download_button(
        "Download this selection as CSV",
        view[show].to_csv(index=False).encode("utf-8"),
        file_name="parcl_segment_selection.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Model: K-Means, K=4, ten standardised client-level behaviour features. "
    "Silhouette 0.228, Davies-Bouldin 1.27, Calinski-Harabasz 619 - all "
    "three indices independently select K=4. Segment labels describe "
    "observed purchasing behaviour only; the dataset contains no income or "
    "wealth field, so no affluence is inferred."
)
