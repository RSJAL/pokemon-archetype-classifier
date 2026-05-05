"""
app.py — Pokémon Archetype Explorer
Streamlit dashboard: pick a Pokémon, see its archetype + nearest neighbours
on an interactive radar chart.

Run with:  streamlit run app.py
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# ── resolve paths relative to this file ──────────────────────────────────────
HERE = Path(__file__).parent

STAT_COLS   = ["HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed"]
STAT_LABELS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

ARCHETYPE_COLORS = {
    "HP Tank":       "#4FC3F7",
    "Below Average": "#90A4AE",
    "All-Rounder":   "#81C784",
    "Sp. Fighter":   "#CE93D8",
    "Physical Wall": "#FFB74D",
}

TYPE_COLORS = {
    "Normal":   "#A8A878", "Fire":     "#F08030", "Water":    "#6890F0",
    "Electric": "#F8D030", "Grass":    "#78C850", "Ice":      "#98D8D8",
    "Fighting": "#C03028", "Poison":   "#A040A0", "Ground":   "#E0C068",
    "Flying":   "#A890F0", "Psychic":  "#F85888", "Bug":      "#A8B820",
    "Rock":     "#B8A038", "Ghost":    "#705898", "Dragon":   "#7038F8",
    "Dark":     "#705848", "Steel":    "#B8B8D0", "Fairy":    "#EE99AC",
}

# ── data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(HERE / "pokedex_clustered.csv")
    scaler = joblib.load(HERE / "scaler.pkl")
    X = scaler.transform(df[STAT_COLS].values)
    return df, scaler, X


@st.cache_resource
def build_nn(_X):
    nn = NearestNeighbors(metric="euclidean")
    nn.fit(_X)
    return nn


# ── radar chart (Plotly) ──────────────────────────────────────────────────────
def make_radar(rows, selected_name):
    """
    rows : list of pd.Series (selected first, then neighbours)
    Returns a Plotly figure with one polar trace per Pokémon.
    """
    categories = STAT_LABELS + [STAT_LABELS[0]]  # close the polygon

    fig = go.Figure()
    for row in rows:
        is_selected = row["Name"] == selected_name
        values = [int(row[c]) for c in STAT_COLS] + [int(row[STAT_COLS[0]])]
        color  = ARCHETYPE_COLORS.get(row["Archetype"], "#888888")

        hover = (
            f"<b>{row['Name']}</b><br>"
            + "<br>".join(f"{lbl}: {int(row[c])}" for lbl, c in zip(STAT_LABELS, STAT_COLS))
            + f"<br>Total: {int(row['Total'])}"
            + f"<br>Archetype: {row['Archetype']}"
            + "<extra></extra>"
        )

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor=color,
            opacity=0.75 if is_selected else 0.18,
            line=dict(
                color=color,
                width=3.5 if is_selected else 1.5,
                dash="solid" if is_selected else "dot",
            ),
            name=row["Name"],
            hovertemplate=hover,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor="#f7f7f7",
            radialaxis=dict(
                visible=True,
                range=[0, 175],
                tickvals=[40, 80, 120, 160],
                tickfont=dict(size=9, color="#aaa"),
                gridcolor="#ddd",
                linecolor="#ddd",
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color="#333"),
                gridcolor="#ddd",
                linecolor="#ddd",
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.30,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        margin=dict(l=70, r=70, t=20, b=100),
        height=480,
        paper_bgcolor="white",
    )
    return fig


# ── stat bar helper ───────────────────────────────────────────────────────────
def stat_bar_html(label, value, max_val, color):
    pct = min(value / max_val * 100, 100)
    return (
        f'<div style="margin:4px 0;display:flex;align-items:center;gap:6px">'
        f'  <span style="width:72px;font-size:12px;color:#555">{label}</span>'
        f'  <span style="width:32px;font-size:12px;font-weight:700;text-align:right">{value}</span>'
        f'  <span style="flex:1;background:#eee;border-radius:4px;height:11px;display:inline-block">'
        f'    <span style="display:block;background:{color};border-radius:4px;'
        f'           width:{pct}%;height:11px"></span>'
        f'  </span>'
        f'</div>'
    )


def type_badge(t):
    bg = TYPE_COLORS.get(t, "#888")
    return (
        f'<span style="background:{bg};color:white;padding:3px 11px;'
        f'border-radius:12px;font-size:12px;font-weight:600;margin-right:4px">{t}</span>'
    )


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pokémon Archetype Explorer",
    page_icon="pokeball",
    layout="wide",
)

# ── load data ─────────────────────────────────────────────────────────────────
df, scaler, X = load_data()
nn_model = build_nn(X)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pokémon Archetype Explorer")
    st.caption("Cluster: K-Means (K=5) on base stats. Similarity: Euclidean distance in scaled stat space.")
    st.divider()

    selected_name = st.selectbox(
        "Choose a Pokémon",
        sorted(df["Name"].tolist()),
        index=sorted(df["Name"].tolist()).index("Garchomp"),
    )
    n_neighbours = st.slider("Neighbours to show", 3, 10, 5)
    same_archetype = st.checkbox("Same-archetype neighbours only", value=False)

    st.divider()
    st.markdown("**Archetypes**")
    for name, color in ARCHETYPE_COLORS.items():
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{color};border-radius:50%;margin-right:6px;vertical-align:middle"></span>'
            f'<span style="font-size:13px">{name}</span>',
            unsafe_allow_html=True,
        )

# ── look up selected Pokémon ──────────────────────────────────────────────────
idx      = df[df["Name"] == selected_name].index[0]
selected = df.loc[idx]
x_sel    = X[idx].reshape(1, -1)

# NearestNeighbors — fetch extra to allow filtering + self exclusion
distances, indices = nn_model.kneighbors(x_sel, n_neighbors=n_neighbours + 25)
distances, indices = distances[0], indices[0]

# Remove self
mask      = indices != idx
distances = distances[mask]
indices   = indices[mask]

neighbours_df = df.iloc[indices].copy()
neighbours_df["Distance"] = np.round(distances, 3)

if same_archetype:
    neighbours_df = neighbours_df[neighbours_df["Archetype"] == selected["Archetype"]]

neighbours_df = neighbours_df.head(n_neighbours).reset_index(drop=True)

# ── main layout ───────────────────────────────────────────────────────────────
col_card, col_radar = st.columns([1, 2], gap="large")

# ── left: Pokémon card ────────────────────────────────────────────────────────
with col_card:
    archetype = selected["Archetype"]
    color     = ARCHETYPE_COLORS.get(archetype, "#888")

    st.markdown(f"## {selected_name}")

    # Generation + dex number
    st.markdown(
        f"**#{int(selected['Dex_Number'])}** &nbsp;·&nbsp; Generation {int(selected['Generation'])}",
        unsafe_allow_html=True,
    )

    # Type badges
    types      = str(selected["Type"]).strip().split()
    badge_html = "".join(type_badge(t) for t in types)
    st.markdown(badge_html + "<br>", unsafe_allow_html=True)

    # Archetype pill
    st.markdown(
        f'<div style="background:{color}28;border-left:4px solid {color};'
        f'padding:8px 14px;border-radius:6px;font-weight:700;font-size:15px;margin:8px 0 16px">'
        f'Archetype: {archetype}</div>',
        unsafe_allow_html=True,
    )

    # Stat bars
    st.markdown("**Base Stats**")
    for lbl, col_name in zip(STAT_LABELS, STAT_COLS):
        st.markdown(stat_bar_html(lbl, int(selected[col_name]), 200, color), unsafe_allow_html=True)
    st.markdown(stat_bar_html("Total", int(selected["Total"]), 700, "#888"), unsafe_allow_html=True)

# ── right: radar chart ────────────────────────────────────────────────────────
with col_radar:
    st.markdown(f"#### {selected_name} vs nearest neighbours")
    rows_to_plot = [selected] + [neighbours_df.iloc[i] for i in range(len(neighbours_df))]
    st.plotly_chart(make_radar(rows_to_plot, selected_name), use_container_width=True)

# ── neighbour table ───────────────────────────────────────────────────────────
st.divider()
st.markdown(f"#### {n_neighbours} Nearest Neighbours (Euclidean distance in scaled stat space)")

# Build display dataframe with archetype-coloured name cells
display_df = neighbours_df[
    ["Name", "Type", "Generation", "Archetype"] + STAT_COLS + ["Total", "Distance"]
].copy()
display_df["Generation"] = display_df["Generation"].astype(int)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Distance": st.column_config.NumberColumn(format="%.3f"),
        "HP":       st.column_config.ProgressColumn("HP",      min_value=0, max_value=200, format="%d"),
        "Attack":   st.column_config.ProgressColumn("Attack",  min_value=0, max_value=200, format="%d"),
        "Defense":  st.column_config.ProgressColumn("Defense", min_value=0, max_value=200, format="%d"),
        "Sp_Atk":   st.column_config.ProgressColumn("Sp. Atk",min_value=0, max_value=200, format="%d"),
        "Sp_Def":   st.column_config.ProgressColumn("Sp. Def",min_value=0, max_value=200, format="%d"),
        "Speed":    st.column_config.ProgressColumn("Speed",   min_value=0, max_value=200, format="%d"),
        "Total":    st.column_config.ProgressColumn("Total",   min_value=0, max_value=700, format="%d"),
    },
)
