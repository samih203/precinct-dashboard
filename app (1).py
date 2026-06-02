"""
Central Florida Precinct Intelligence Dashboard — Streamlit version
Run: streamlit run app.py
"""

import os, sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "models"))
sys.path.insert(0, os.path.join(BASE, "data"))

# ── Bootstrap data + models — fully in-memory, no disk writes ─────────────────
@st.cache_data
def load_data():
    # Use pregenerated CSV if it was committed to the repo, else generate in memory
    data_path = os.path.join(BASE, "data", "precincts.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    from generate_data import generate_precincts, compute_features
    df = generate_precincts()
    df = compute_features(df)
    return df  # never written to disk

@st.cache_resource
def load_models(_df):
    # Always train in memory — avoids any pickle file writes on read-only filesystems
    from predictor import train_models
    tm, mm, metrics, _, _ = train_models(_df)
    return tm, mm, metrics

df = load_data()
tm, mm, metrics = load_models(df)  # underscore arg name tells Streamlit not to hash the df

COUNTIES   = sorted(df["county"].unique().tolist())
PTYPES     = sorted(df["precinct_type"].unique().tolist())
ELECTIONS  = [2016, 2018, 2020, 2022, 2024]

# ── Theme helpers ──────────────────────────────────────────────────────────────
THEME = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0", family="Inter, sans-serif", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#334155"),
    legend=dict(bgcolor="#0f172a", bordercolor="#334155"),
)

COUNTY_COLORS = ["#f97316","#3b82f6","#10b981","#8b5cf6","#f59e0b"]

def styled(fig):
    fig.update_layout(**THEME)
    return fig

def margin_color(m):
    if m >  0.20: return "#1d4ed8"
    if m >  0.10: return "#3b82f6"
    if m >  0.02: return "#93c5fd"
    if m > -0.02: return "#c4b5fd"
    if m > -0.10: return "#fca5a5"
    if m > -0.20: return "#ef4444"
    return "#b91c1c"

def party_label(m):
    if m >  0.05: return "Safe D"
    if m >  0.01: return "Lean D"
    if m > -0.01: return "Tossup"
    if m > -0.05: return "Lean R"
    return "Safe R"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Central Florida Precinct Dashboard",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: #0f172a; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background: #1e293b; }
    .block-container { padding-top: 1.5rem; }
    .metric-container { background: #1e293b; border-radius: 10px; padding: 14px 18px; border: 1px solid #334155; }
    h1, h2, h3 { color: #f1f5f9; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #e2e8f0; border-bottom: 2px solid #3b82f6; }
    div[data-testid="stMetricValue"] { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #64748b; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .stDataFrame { background: #1e293b; }
    label { color: #94a3b8 !important; }
    .stSelectbox > div, .stMultiSelect > div { background: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 🗺️ Central Florida Precinct Intelligence")
    st.markdown("<p style='color:#64748b;margin-top:-12px'>Congressional District Analysis · Turnout Modeling · Margin Prediction</p>", unsafe_allow_html=True)
with col_h2:
    st.markdown(f"""
    <div style='text-align:right;padding-top:8px'>
        <span style='background:#10b981;color:#fff;padding:4px 12px;border-radius:999px;font-size:11px;margin-right:6px'>
            Turnout R² {metrics['turnout_r2']:.2f}
        </span>
        <span style='background:#3b82f6;color:#fff;padding:4px 12px;border-radius:999px;font-size:11px'>
            Margin R² {metrics['margin_r2']:.2f}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Precinct Map", "🤖 ML Predictions", "📈 Historical Trends", "📊 District Summary"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRECINCT MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    sb = st.sidebar
    sb.markdown("### 🗺️ Map Controls")

    year = sb.selectbox("Election Year", ELECTIONS, index=4)
    metric = sb.selectbox("Color By", {
        "margin": "D/R Margin",
        "turnout": "Turnout %",
        "dem_share": "Dem Vote Share",
        "votes_cast": "Votes Cast",
    }.keys(), format_func=lambda x: {
        "margin": "D/R Margin",
        "turnout": "Turnout %",
        "dem_share": "Dem Vote Share",
        "votes_cast": "Votes Cast",
    }[x])

    counties_sel = sb.multiselect("Counties", COUNTIES, default=COUNTIES)
    types_sel    = sb.multiselect("Precinct Types", PTYPES, default=[])
    min_comp     = sb.slider("Min Competitiveness", 0.0, 1.0, 0.0, 0.1)

    dff = df[df["county"].isin(counties_sel or COUNTIES)].copy()
    if types_sel:
        dff = dff[dff["precinct_type"].isin(types_sel)]
    dff = dff[dff["competitiveness"] >= min_comp]

    col_name = f"{year}_{metric}"
    dff["_val"] = dff[col_name]
    dff["_label"] = dff[col_name].apply(
        lambda v: f"{v:+.1%}" if metric == "margin"
        else f"{v:.1%}" if metric in ("turnout", "dem_share")
        else f"{int(v):,}"
    )
    dff["_party"] = dff[f"{year}_margin"].apply(party_label)
    dff["_color"] = dff[f"{year}_margin"].apply(margin_color)

    if metric == "margin":
        color_scale = [[0,"#b91c1c"],[0.45,"#fca5a5"],[0.499,"#c4b5fd"],[0.501,"#c4b5fd"],[0.55,"#93c5fd"],[1,"#1d4ed8"]]
        cmin, cmax = -1, 1
    elif metric == "turnout":
        color_scale = "Blues"
        cmin, cmax = 0.3, 0.9
    else:
        color_scale = "RdBu"
        cmin, cmax = None, None

    fig_map = px.scatter_mapbox(
        dff,
        lat="latitude", lon="longitude",
        color="_val",
        size="registered_voters",
        size_max=18,
        hover_name="precinct_name",
        hover_data={
            "county": True,
            "precinct_type": True,
            "registered_voters": ":,",
            "_label": True,
            "_party": True,
            "_val": False,
        },
        color_continuous_scale=color_scale,
        range_color=[cmin, cmax] if cmin is not None else None,
        mapbox_style="carto-darkmatter",
        zoom=8.2,
        center={"lat": 28.55, "lon": -81.20},
        opacity=0.85,
        labels={"_label": metric.replace("_"," ").title(), "_party": "Rating"},
    )
    fig_map.update_layout(
        paper_bgcolor="#0f172a",
        margin=dict(l=0,r=0,t=0,b=0),
        height=560,
        coloraxis_colorbar=dict(thickness=12, len=0.6),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Precinct detail on click via selectbox
    st.markdown("#### Precinct Detail")
    selected = st.selectbox(
        "Select a precinct to inspect",
        ["— select —"] + sorted(dff["precinct_name"].tolist()),
    )
    if selected != "— select —":
        r = df[df["precinct_name"] == selected].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registered Voters", f"{int(r['registered_voters']):,}")
        c2.metric("Dem Registration", f"{r['pct_dem_registered']:.1%}")
        c3.metric("Rep Registration", f"{r['pct_rep_registered']:.1%}")
        c4.metric("Competitiveness", f"{r['competitiveness']:.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Hispanic %", f"{r['pct_hispanic']:.1%}")
        c6.metric("Black %", f"{r['pct_black']:.1%}")
        c7.metric("Senior 65+ %", f"{r['pct_senior']:.1%}")
        c8.metric("Median Income", f"${r['median_income']:,}")

        st.markdown("**Election History**")
        hist = []
        for y in ELECTIONS:
            hist.append({
                "Year": y,
                "Turnout": f"{r[f'{y}_turnout']:.1%}",
                "Dem Votes": f"{int(r[f'{y}_dem_votes']):,}",
                "Rep Votes": f"{int(r[f'{y}_rep_votes']):,}",
                "Margin": f"{r[f'{y}_margin']:+.1%}",
                "Rating": party_label(r[f"{y}_margin"]),
            })
        st.dataframe(pd.DataFrame(hist), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    from predictor import predict

    st.markdown("### Scenario Controls")
    pc1, pc2, pc3 = st.columns(3)
    nat_env   = pc1.slider("National Environment Shift (pp)", -10, 10, 0, 1,
                            help="+3 = Dem wave, -3 = Rep wave")
    gotv      = pc2.slider("GOTV / Ground Game Boost (pp turnout)", 0, 10, 0, 1,
                            help="Simulates canvassing impact")
    pred_cnty = pc3.selectbox("Target County", ["All"] + COUNTIES)

    run = st.button("Run Predictions →", type="primary")

    if run or "pred_df" not in st.session_state:
        dff_p = df if pred_cnty == "All" else df[df["county"] == pred_cnty]
        preds = predict(dff_p, tm, mm,
                        national_env=nat_env / 100,
                        ground_game_boost=gotv / 100)
        merged = dff_p.merge(preds, on="precinct_id")
        st.session_state["pred_df"] = merged

    merged = st.session_state["pred_df"]
    total_votes = int(merged["pred_votes_cast"].sum())
    dem_votes   = int(merged["pred_dem_votes"].sum())
    rep_votes   = int(merged["pred_rep_votes"].sum())
    overall     = (dem_votes - rep_votes) / total_votes if total_votes else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Total Votes", f"{total_votes:,}")
    m2.metric("Predicted Dem Margin", f"{overall:+.1%}",
              delta=f"{nat_env:+d}pp env" if nat_env else None)
    m3.metric("Avg Predicted Turnout", f"{merged['pred_turnout'].mean():.1%}",
              delta=f"+{gotv}pp GOTV" if gotv else None)
    m4.metric("Precincts Modeled", str(len(merged)))

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        fig_scatter = px.scatter(
            merged, x="pred_turnout", y="pred_margin",
            color="county", size="registered_voters", size_max=20,
            hover_name="precinct_name",
            hover_data={"pred_turnout":":.1%","pred_margin":":.1%","pred_votes_cast":":,"},
            color_discrete_sequence=COUNTY_COLORS,
            title="Predicted Turnout vs. Margin",
            labels={"pred_turnout":"Predicted Turnout","pred_margin":"Predicted Margin"},
            height=420,
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="#475569",
                               annotation_text="Even")
        styled(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        by_type = merged.groupby("precinct_type").agg(
            avg_turnout=("pred_turnout","mean"),
            avg_margin=("pred_margin","mean"),
        ).reset_index().sort_values("avg_margin")

        fig_type = go.Figure()
        fig_type.add_trace(go.Bar(
            x=by_type["avg_turnout"], y=by_type["precinct_type"],
            orientation="h", name="Turnout", marker_color="#10b981",
        ))
        fig_type.update_layout(
            title="Avg Turnout by Precinct Type",
            xaxis=dict(tickformat=".0%"),
            height=420, **THEME,
        )
        st.plotly_chart(fig_type, use_container_width=True)

    # Priority table
    st.markdown("### 🎯 High-Priority Precincts")
    st.caption("Competitive precincts (±10pp margin) with most registered voters — best targets for canvassing")
    priority = merged[merged["pred_margin"].abs() < 0.10].copy()
    priority = priority.sort_values("registered_voters", ascending=False).head(25)
    priority["Rating"] = priority["pred_margin"].apply(party_label)
    display_cols = {
        "precinct_name": "Precinct",
        "county": "County",
        "precinct_type": "Type",
        "registered_voters": "Reg. Voters",
        "pred_turnout": "Pred. Turnout",
        "pred_margin": "Pred. Margin",
        "pred_votes_cast": "Pred. Votes",
        "Rating": "Rating",
    }
    out = priority[list(display_cols.keys())].rename(columns=display_cols)
    out["Pred. Turnout"] = out["Pred. Turnout"].apply(lambda x: f"{x:.1%}")
    out["Pred. Margin"]  = out["Pred. Margin"].apply(lambda x: f"{x:+.1%}")
    out["Reg. Voters"]   = out["Reg. Voters"].apply(lambda x: f"{int(x):,}")
    out["Pred. Votes"]   = out["Pred. Votes"].apply(lambda x: f"{int(x):,}")
    st.dataframe(out, hide_index=True, use_container_width=True)

    csv = priority.to_csv(index=False).encode()
    st.download_button("⬇️ Export Priority Precincts as CSV", csv,
                        "priority_precincts.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORICAL TRENDS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Historical Trends 2016–2024")
    tc1, tc2 = st.columns(2)
    t_county = tc1.selectbox("County", ["All"] + COUNTIES, key="tc")
    t_type   = tc2.selectbox("Precinct Type", ["All"] + PTYPES, key="tt")

    dft = df.copy()
    if t_county != "All": dft = dft[dft["county"] == t_county]
    if t_type   != "All": dft = dft[dft["precinct_type"] == t_type]

    rows = []
    for y in ELECTIONS:
        for c in dft["county"].unique():
            cdf = dft[dft["county"] == c]
            rows.append({"Year": y, "County": c,
                          "Avg Turnout": cdf[f"{y}_turnout"].mean(),
                          "Avg Margin":  cdf[f"{y}_margin"].mean()})
    tdf = pd.DataFrame(rows)
    cmap = {c: v for c, v in zip(COUNTIES, COUNTY_COLORS)}

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(tdf, x="Year", y="Avg Turnout", color="County",
                        markers=True, title="Average Turnout by County",
                        color_discrete_map=cmap, height=340)
        fig1.update_yaxes(tickformat=".0%")
        styled(fig1)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.line(tdf, x="Year", y="Avg Margin", color="County",
                        markers=True, title="Average Dem Margin by County",
                        color_discrete_map=cmap, height=340)
        fig2.add_hline(y=0, line_dash="dash", line_color="#475569")
        fig2.update_yaxes(tickformat="+.0%")
        styled(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(dft, x="2020_turnout", y="2024_turnout", color="county",
                           size="registered_voters", size_max=16,
                           hover_name="precinct_name",
                           title="Turnout: 2020 vs 2024",
                           color_discrete_map=cmap, height=340)
        fig3.add_shape(type="line", x0=0.2,y0=0.2,x1=0.95,y1=0.95,
                        line=dict(dash="dash",color="#475569"))
        fig3.update_xaxes(tickformat=".0%")
        fig3.update_yaxes(tickformat=".0%")
        styled(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        dft["swing"] = dft["2024_margin"] - dft["2020_margin"]
        swing = dft.groupby("precinct_type")["swing"].mean().reset_index().sort_values("swing")
        fig4 = go.Figure(go.Bar(
            x=swing["swing"], y=swing["precinct_type"],
            orientation="h",
            marker_color=["#3b82f6" if v > 0 else "#ef4444" for v in swing["swing"]],
        ))
        fig4.update_layout(title="Dem Swing by Type (2020→2024)",
                            xaxis=dict(tickformat="+.0%"), height=340, **THEME)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DISTRICT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### District Overview")

    total_reg   = int(df["registered_voters"].sum())
    avg_turnout = df["2024_turnout"].mean()
    avg_margin  = df["2024_margin"].mean()
    n_tossup    = int((df["competitiveness"] > 0.8).sum())
    n_safe_d    = int((df["2024_margin"] > 0.10).sum())
    n_safe_r    = int((df["2024_margin"] < -0.10).sum())

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Total Registered Voters", f"{total_reg:,}", f"{len(df)} precincts")
    s2.metric("2024 Avg Turnout",  f"{avg_turnout:.1%}")
    s3.metric("2024 Avg Margin",   f"{avg_margin:+.1%}")
    s4.metric("Tossup Precincts",  str(n_tossup), f"Safe D: {n_safe_d} · Safe R: {n_safe_r}")

    st.markdown("---")
    st.markdown("#### County Breakdown (2024)")
    county_tbl = df.groupby("county").agg(
        Precincts=("precinct_id","count"),
        Voters=("registered_voters","sum"),
        Turnout=("2024_turnout","mean"),
        Margin=("2024_margin","mean"),
    ).reset_index().rename(columns={"county":"County"})
    county_tbl["Voters"]  = county_tbl["Voters"].apply(lambda x: f"{int(x):,}")
    county_tbl["Turnout"] = county_tbl["Turnout"].apply(lambda x: f"{x:.1%}")
    county_tbl["Margin"]  = county_tbl["Margin"].apply(lambda x: f"{x:+.1%}")
    st.dataframe(county_tbl, hide_index=True, use_container_width=True)

    st.markdown("---")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pt_dist = df.groupby("precinct_type").size().reset_index(name="count")
        fig_pie = px.pie(pt_dist, names="precinct_type", values="count",
                          hole=0.5, title="Precinct Type Distribution",
                          color_discrete_sequence=px.colors.qualitative.Bold, height=360)
        styled(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_p2:
        reg_bal = df.groupby("county")[["pct_dem_registered","pct_rep_registered"]].mean().reset_index()
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Bar(name="Dem", x=reg_bal["county"],
                                  y=reg_bal["pct_dem_registered"], marker_color="#3b82f6"))
        fig_reg.add_trace(go.Bar(name="Rep", x=reg_bal["county"],
                                  y=reg_bal["pct_rep_registered"], marker_color="#ef4444"))
        fig_reg.update_layout(title="Party Registration by County", barmode="group",
                               yaxis=dict(tickformat=".0%"), height=360, **THEME)
        st.plotly_chart(fig_reg, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Export Full Dataset")
    st.download_button(
        "⬇️ Download All Precinct Data as CSV",
        df.to_csv(index=False).encode(),
        "central_florida_precincts.csv",
        "text/csv",
    )
