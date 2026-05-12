"""
app.py — Pokémon Archetype Explorer
Two views:
  1. Neighbour Web  — selected Pokémon in the centre, 5 nearest neighbours around it (with images)
  2. Radar & Stats  — original radar chart + stat bars + neighbour table

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
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
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

# ── name helpers ──────────────────────────────────────────────────────────────
def display_name(csv_name: str) -> str:
    """Convert raw CSV name to a clean UI label."""
    if "  " in csv_name:
        base, form = csv_name.split("  ", 1)
        form = form.strip()
        # Regional forms: "Alolan X", "Galarian X" etc. → use form name directly
        if form.split()[0] in ("Alolan", "Galarian", "Hisuian", "Paldean"):
            return form
        return f"{base.strip()} ({form})"
    return csv_name


# ── data loading ──────────────────────────────────────────────────────────────
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


@st.cache_data
def compute_pca(_X):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(_X)
    var = pca.explained_variance_ratio_ * 100
    return coords, var


@st.cache_data
def load_image_map():
    path = HERE / "image_map.csv"
    if not path.exists():
        return {}
    df_map = pd.read_csv(path)
    return dict(zip(df_map["csv_name"], df_map["path"].fillna("")))


def get_image(csv_name: str, image_map: dict):
    """Load a Pokémon image as a numpy RGBA array, or None if unavailable."""
    path_str = image_map.get(csv_name, "")
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return np.array(Image.open(p).convert("RGBA"))
    except Exception:
        return None


# ── neighbour web figure ──────────────────────────────────────────────────────
# Slot positions: index 0 = centre (selected), 1-5 = neighbours
_POSITIONS = [
    (0.0,   0.11),   # centre
    (-1.32, 1.155),  # top-left
    (0.0,   1.65),   # top-centre
    (1.32,  1.155),  # top-right
    (-1.21, -0.935), # bottom-left
    (1.21,  -0.935), # bottom-right
]


def make_neighbour_web(selected_row, neighbours_df, image_map):
    n = min(5, len(neighbours_df))
    pokemon_list = [selected_row] + [neighbours_df.iloc[i] for i in range(n)]

    BG = "#0e1117"

    fig, ax = plt.subplots(figsize=(4.86, 4.68))
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.70, 1.82)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    cx, cy = _POSITIONS[0]

    # Pokéball watermark behind everything
    ball_r = 1.705
    ball_alpha = 0.07
    # Top half (red)
    from matplotlib.patches import Wedge, Circle as MplCircle
    ax.add_patch(Wedge((cx, cy), ball_r, 0, 180,
                       facecolor="#cc2200", alpha=ball_alpha, zorder=1, edgecolor="none"))
    # Bottom half (white)
    ax.add_patch(Wedge((cx, cy), ball_r, 180, 360,
                       facecolor="#dddddd", alpha=ball_alpha, zorder=1, edgecolor="none"))
    # Outer ring
    ax.add_patch(MplCircle((cx, cy), ball_r, fill=False,
                            edgecolor="#888888", lw=1.5, alpha=ball_alpha * 2.5, zorder=1))
    # Centre band
    ax.plot([cx - ball_r, cx + ball_r], [cy, cy],
            color="#888888", lw=2.5, alpha=ball_alpha * 2.5, zorder=1)
    # Centre button
    ax.add_patch(MplCircle((cx, cy), 0.198,
                            facecolor="#aaaaaa", edgecolor="#888888",
                            lw=1.5, alpha=ball_alpha * 3, zorder=1))
    ax.add_patch(MplCircle((cx, cy), 0.11,
                            facecolor="#ffffff", edgecolor="none",
                            alpha=ball_alpha * 3, zorder=1))

    # Concentric rings from centre for spatial grounding
    for r, alpha in [(0.605, 0.12), (1.155, 0.18), (1.705, 0.12)]:
        circle = plt.Circle((cx, cy), r, color="#7a8aaa",
                             fill=False, lw=0.8, alpha=alpha, zorder=1,
                             linestyle="--")
        ax.add_patch(circle)

    # Dashed connector lines from centre to each neighbour
    for i in range(1, n + 1):
        nx, ny = _POSITIONS[i]
        dist = neighbours_df.iloc[i - 1].get("Distance", None)
        ax.plot([cx, nx], [cy, ny], color="#3a3a4a", lw=1.2,
                linestyle="--", zorder=1)

    # Place images + labels
    for i, poke_row in enumerate(pokemon_list):
        px, py = _POSITIONS[i]
        is_centre = i == 0
        zoom = 0.259 if is_centre else 0.160

        img_arr = get_image(poke_row["Name"], image_map)
        if img_arr is not None:
            imgbox = OffsetImage(img_arr, zoom=zoom)
            ab = AnnotationBbox(imgbox, (px, py), frameon=False, zorder=3)
            ax.add_artist(ab)
        else:
            # Placeholder circle
            r = 0.38 if is_centre else 0.24
            circle = plt.Circle((px, py), r, color="#2e3340",
                                 ec="#252a35", lw=1.5, zorder=3)
            ax.add_patch(circle)
            ax.text(px, py, "?", ha="center", va="center",
                    fontsize=20 if is_centre else 14,
                    color="#cccccc", zorder=4)

        # Name label
        name = display_name(poke_row["Name"])
        label_y = py - (0.462 if is_centre else 0.286)
        ax.text(px, label_y, name,
                ha="center", va="top",
                fontsize=11 if is_centre else 8.5,
                fontweight="bold" if is_centre else "normal",
                color="#ffffff", zorder=4)

    plt.tight_layout(pad=0.5)
    return fig


# ── radar chart ───────────────────────────────────────────────────────────────
def make_radar(rows, selected_name):
    categories = STAT_LABELS + [STAT_LABELS[0]]
    fig = go.Figure()
    for row in rows:
        is_selected = row["Name"] == selected_name
        values = [int(row[c]) for c in STAT_COLS] + [int(row[STAT_COLS[0]])]
        color  = ARCHETYPE_COLORS.get(row["Archetype"], "#888888")
        hover = (
            f"<b>{display_name(row['Name'])}</b><br>"
            + "<br>".join(f"{lbl}: {int(row[c])}" for lbl, c in zip(STAT_LABELS, STAT_COLS))
            + f"<br>Total: {int(row['Total'])}"
            + f"<br>Archetype: {row['Archetype']}"
            + "<extra></extra>"
        )
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories,
            fill="toself", fillcolor=color,
            opacity=0.75 if is_selected else 0.18,
            line=dict(color=color,
                      width=3.5 if is_selected else 1.5,
                      dash="solid" if is_selected else "dot"),
            name=display_name(row["Name"]),
            hovertemplate=hover,
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#f7f7f7",
            radialaxis=dict(visible=True, range=[0, 175],
                            tickvals=[40, 80, 120, 160],
                            tickfont=dict(size=9, color="#aaa"),
                            gridcolor="#ddd", linecolor="#ddd"),
            angularaxis=dict(tickfont=dict(size=13, color="#333"),
                             gridcolor="#ddd", linecolor="#ddd"),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.30,
                    xanchor="center", x=0.5, font=dict(size=12)),
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
        f'  <span style="width:72px;font-size:12px;color:#cccccc">{label}</span>'
        f'  <span style="width:32px;font-size:12px;font-weight:700;text-align:right">{value}</span>'
        f'  <span style="flex:1;background:#eee;border-radius:4px;height:11px;display:inline-block">'
        f'    <span style="display:block;background:{color};border-radius:4px;'
        f'           width:{pct}%;height:11px"></span>'
        f'  </span>'
        f'</div>'
    )


def comparison_bar_html(label, val_cmp, val_ref, color_cmp, color_ref, max_val=200):
    """Dual stat bar: reference (dimmed) behind, comparison on top."""
    pct_cmp = min(val_cmp / max_val * 100, 100)
    pct_ref = min(val_ref / max_val * 100, 100)
    return (
        f'<div style="margin:5px 0;display:flex;align-items:center;gap:6px">'
        f'  <span style="width:72px;font-size:12px;color:#cccccc">{label}</span>'
        f'  <span style="width:32px;font-size:12px;font-weight:700;text-align:right;color:{color_cmp}">{val_cmp}</span>'
        f'  <span style="flex:1;position:relative;background:#1e1e2a;border-radius:4px;height:12px;display:inline-block">'
        f'    <span style="display:block;position:absolute;background:{color_ref};border-radius:4px;'
        f'           width:{pct_ref}%;height:12px;opacity:0.28"></span>'
        f'    <span style="display:block;position:absolute;background:{color_cmp};border-radius:4px;'
        f'           width:{pct_cmp}%;height:12px;opacity:0.85"></span>'
        f'    <span style="display:block;position:absolute;left:{pct_ref}%;top:-2px;'
        f'           width:2px;height:16px;background:white;opacity:0.85;'
        f'           transform:translateX(-1px);border-radius:1px"></span>'
        f'  </span>'
        f'  <span style="width:32px;font-size:11px;text-align:left;color:#aaaaaa">{val_ref}</span>'
        f'</div>'
    )


def type_badge(t):
    bg = TYPE_COLORS.get(t, "#888")
    return (
        f'<span style="background:{bg};color:white;padding:3px 11px;'
        f'border-radius:12px;font-size:12px;font-weight:600;margin-right:4px">{t}</span>'
    )


# ── page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pokémon Archetype Explorer",
    page_icon="pokeball",
    layout="wide",
)

df, scaler, X = load_data()
nn_model  = build_nn(X)
image_map = load_image_map()

# Build display↔csv name maps (display names are unique)
all_csv_names  = sorted(df["Name"].tolist())
all_disp_names = [display_name(n) for n in all_csv_names]
disp_to_csv    = dict(zip(all_disp_names, all_csv_names))

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    default_disp = display_name("Garchomp")
    selected_disp = st.selectbox(
        "Choose a Pokémon",
        all_disp_names,
        index=all_disp_names.index(default_disp),
    )
    selected_name = disp_to_csv[selected_disp]

    n_neighbours  = 5
    same_archetype = False

    _sel = df.loc[df["Name"] == selected_name].iloc[0]
    sel_color = ARCHETYPE_COLORS.get(_sel["Archetype"], "#888")

    # Dex number + generation
    st.markdown(
        f"**#{int(_sel['Dex_Number'])}** &nbsp;·&nbsp; Generation {int(_sel['Generation'])}",
        unsafe_allow_html=True,
    )

    # Type badges
    types = str(_sel["Type"]).strip().split()
    st.markdown("".join(type_badge(t) for t in types) + "<br>", unsafe_allow_html=True)

    # Archetype pill
    st.markdown(
        f'<div style="background:{sel_color}28;border-left:4px solid {sel_color};'
        f'padding:6px 12px;border-radius:6px;font-weight:700;font-size:13px;margin:8px 0 12px">'
        f'Archetype: {_sel["Archetype"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Base Stats**")
    for lbl, col_name in zip(STAT_LABELS, STAT_COLS):
        st.markdown(stat_bar_html(lbl, int(_sel[col_name]), 200, sel_color),
                    unsafe_allow_html=True)
    st.markdown(stat_bar_html("Total", int(_sel["Total"]), 700, "#888"),
                unsafe_allow_html=True)

    st.divider()
    st.markdown("**Archetypes**")
    for arch_name, arch_color in ARCHETYPE_COLORS.items():
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{arch_color};border-radius:50%;margin-right:6px;vertical-align:middle"></span>'
            f'<span style="font-size:13px">{arch_name}</span>',
            unsafe_allow_html=True,
        )

    if not image_map:
        st.divider()
        st.warning("No images found. Run `python download_images.py` to download artwork.")

# ── look up selected Pokémon + neighbours ─────────────────────────────────────
idx      = df[df["Name"] == selected_name].index[0]
selected = df.loc[idx]
x_sel    = X[idx].reshape(1, -1)

distances, indices = nn_model.kneighbors(x_sel, n_neighbors=n_neighbours + 25)
distances, indices = distances[0], indices[0]

mask      = indices != idx
distances = distances[mask]
indices   = indices[mask]

neighbours_df = df.iloc[indices].copy()
neighbours_df["Distance"] = np.round(distances, 3)

if same_archetype:
    neighbours_df = neighbours_df[
        neighbours_df["Archetype"] == selected["Archetype"]
    ]

neighbours_df = neighbours_df.head(n_neighbours).reset_index(drop=True)

# ── tabs ──────────────────────────────────────────────────────────────────────
st.title("Pokémon Archetype Explorer")
st.caption("K-Means (K=5) on base stats · Similarity: Euclidean distance in scaled stat space.")

tab_web, tab_radar = st.tabs(["Neighbour Web", "Radar & Stats"])

# ── TAB 1: Neighbour Web ──────────────────────────────────────────────────────
with tab_web:
    import io
    col_web, col_compare = st.columns([1, 1])

    # Left: neighbour web
    with col_web:
        neighbours_5 = neighbours_df.head(5).reset_index(drop=True)
        fig_web = make_neighbour_web(selected, neighbours_5, image_map)
        buf = io.BytesIO()
        fig_web.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                        facecolor="#0e1117")
        buf.seek(0)
        plt.close(fig_web)
        st.image(buf, width=655)

    # Right: comparison panel
    with col_compare:
        compare_disp = st.selectbox(
            "Compare with",
            all_disp_names,
            index=0,
            key="compare_select",
        )
        compare_name = disp_to_csv[compare_disp]
        _cmp = df.loc[df["Name"] == compare_name].iloc[0]
        cmp_idx   = df[df["Name"] == compare_name].index[0]
        cmp_color = ARCHETYPE_COLORS.get(_cmp["Archetype"], "#888")
        sel_color_ref = ARCHETYPE_COLORS.get(selected["Archetype"], "#888")

        # Pokémon image
        cmp_img_path = image_map.get(compare_name, "")
        if cmp_img_path and Path(cmp_img_path).exists():
            st.image(cmp_img_path, width=130)

        # Dex # and generation
        st.markdown(
            f"**#{int(_cmp['Dex_Number'])}** &nbsp;·&nbsp; Generation {int(_cmp['Generation'])}",
            unsafe_allow_html=True,
        )

        # Type badges
        cmp_types = str(_cmp["Type"]).strip().split()
        st.markdown("".join(type_badge(t) for t in cmp_types) + "<br>", unsafe_allow_html=True)

        # Archetype pill
        st.markdown(
            f'<div style="background:{cmp_color}28;border-left:4px solid {cmp_color};'
            f'padding:6px 12px;border-radius:6px;font-weight:700;font-size:13px;margin:8px 0 8px">'
            f'Archetype: {_cmp["Archetype"]}</div>',
            unsafe_allow_html=True,
        )

        # Euclidean distance to selected Pokémon
        dist_to_sel = float(np.linalg.norm(X[cmp_idx] - X[idx]))
        st.markdown(
            f'<div style="font-size:13px;color:#aaa;margin-bottom:10px">'
            f'Distance to <b style="color:white">{selected_disp}</b>: '
            f'<b style="color:{cmp_color}">{dist_to_sel:.3f}</b></div>',
            unsafe_allow_html=True,
        )

        # Stat comparison bars (comparison colour vs selected dimmed)
        st.markdown(
            f'<div style="font-size:12px;color:#cccccc;margin-bottom:4px">'
            f'<span style="color:{cmp_color}">&#9632;</span> {compare_disp} &nbsp;&nbsp;'
            f'<span style="color:{sel_color_ref};opacity:0.5">&#9632;</span> {selected_disp}</div>',
            unsafe_allow_html=True,
        )
        for lbl, col_name in zip(STAT_LABELS, STAT_COLS):
            st.markdown(
                comparison_bar_html(lbl, int(_cmp[col_name]), int(selected[col_name]),
                                    cmp_color, sel_color_ref),
                unsafe_allow_html=True,
            )
        st.markdown(
            comparison_bar_html("Total", int(_cmp["Total"]), int(selected["Total"]),
                                cmp_color, sel_color_ref, max_val=700),
            unsafe_allow_html=True,
        )


# ── TAB 2: PCA Explorer ───────────────────────────────────────────────────────
with tab_radar:
    pca_coords, pca_var = compute_pca(X)
    neighbour_idxs = set(neighbours_df.index.tolist())

    fig_pca = go.Figure()

    for archetype, color in ARCHETYPE_COLORS.items():
        mask = df["Archetype"] == archetype
        sub  = df[mask].copy()
        sub_coords = pca_coords[mask]

        hover = [
            f"<b>{display_name(row['Name'])}</b><br>"
            f"#{int(row['Dex_Number'])} · Gen {int(row['Generation'])}<br>"
            f"Type: {row['Type'].strip()}<br>"
            f"Archetype: {row['Archetype']}<br>"
            f"HP {int(row['HP'])} · Atk {int(row['Attack'])} · Def {int(row['Defense'])}<br>"
            f"SpA {int(row['Sp_Atk'])} · SpD {int(row['Sp_Def'])} · Spe {int(row['Speed'])}<br>"
            f"Total: {int(row['Total'])}"
            for _, row in sub.iterrows()
        ]

        fig_pca.add_trace(go.Scatter(
            x=sub_coords[:, 0], y=sub_coords[:, 1],
            mode="markers",
            name=archetype,
            marker=dict(color=color, size=6, opacity=0.6,
                        line=dict(width=0)),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
        ))

    # Nearest neighbours ring
    nb_coords = pca_coords[neighbours_df.index]
    nb_hover  = [
        f"<b>{display_name(row['Name'])}</b><br>"
        f"#{int(row['Dex_Number'])} · Gen {int(row['Generation'])}<br>"
        f"Type: {row['Type'].strip()}<br>"
        f"Archetype: {row['Archetype']}<br>"
        f"HP {int(row['HP'])} · Atk {int(row['Attack'])} · Def {int(row['Defense'])}<br>"
        f"SpA {int(row['Sp_Atk'])} · SpD {int(row['Sp_Def'])} · Spe {int(row['Speed'])}<br>"
        f"Total: {int(row['Total'])}<br>"
        f"Distance: {row['Distance']:.3f}"
        for _, row in neighbours_df.iterrows()
    ]
    fig_pca.add_trace(go.Scatter(
        x=nb_coords[:, 0], y=nb_coords[:, 1],
        mode="markers",
        name="Neighbours",
        marker=dict(color="white", size=10, opacity=0.9,
                    line=dict(color="white", width=2),
                    symbol="circle-open"),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=nb_hover,
    ))

    # Selected Pokémon star
    fig_pca.add_trace(go.Scatter(
        x=[pca_coords[idx, 0]], y=[pca_coords[idx, 1]],
        mode="markers+text",
        name=selected_disp,
        marker=dict(
            color=ARCHETYPE_COLORS.get(selected["Archetype"], "#fff"),
            size=14, symbol="star",
            line=dict(color="white", width=1.5),
        ),
        text=[selected_disp],
        textposition="top center",
        textfont=dict(color="white", size=12),
        hovertemplate=(
            f"<b>{selected_disp}</b><br>"
            f"#{int(selected['Dex_Number'])} · Gen {int(selected['Generation'])}<br>"
            f"Type: {selected['Type'].strip()}<br>"
            f"Archetype: {selected['Archetype']}<br>"
            f"HP {int(selected['HP'])} · Atk {int(selected['Attack'])} · Def {int(selected['Defense'])}<br>"
            f"SpA {int(selected['Sp_Atk'])} · SpD {int(selected['Sp_Def'])} · Spe {int(selected['Speed'])}<br>"
            f"Total: {int(selected['Total'])}"
            "<extra></extra>"
        ),
    ))

    fig_pca.update_layout(
        xaxis_title=f"PC1 ({pca_var[0]:.1f}% variance)",
        yaxis_title=f"PC2 ({pca_var[1]:.1f}% variance)",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        xaxis=dict(gridcolor="#2a2a3a", zerolinecolor="#2a2a3a"),
        yaxis=dict(gridcolor="#2a2a3a", zerolinecolor="#2a2a3a"),
        height=600,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="closest",
    )

    st.plotly_chart(fig_pca, use_container_width=True)

    display_df = neighbours_df[
        ["Name", "Type", "Generation", "Archetype"] + STAT_COLS + ["Total", "Distance"]
    ].copy()
    display_df["Name"]       = display_df["Name"].apply(display_name)
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
