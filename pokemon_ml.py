"""
pokemon_ml.py
-------------
Unsupervised → Supervised → Visualisation pipeline for fully-evolved Pokemon.

Workflow:
  1. Load pokedex_fe.csv
  2. K-Means clustering on standardised base stats (elbow + silhouette to pick K)
  3. KNN + Decision Tree trained on cluster labels
  4. PCA scatter + radar charts per cluster
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import math

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

STAT_COLS = ["HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed"]
STAT_LABELS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

# =============================================================================
# 1. LOAD DATA
# =============================================================================

df = pd.read_csv("pokedex_fe.csv")
print(f"Loaded {len(df)} fully-evolved Pokemon")

X_raw = df[STAT_COLS].values
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# =============================================================================
# 2. UNSUPERVISED — K-MEANS
# =============================================================================

# --- 2a. Elbow + Silhouette to choose K --------------------------------------
K_RANGE = range(2, 12)
inertias, sil_scores = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(K_RANGE), inertias, "o-", color="steelblue")
axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia"); axes[0].set_title("Elbow Method")
axes[1].plot(list(K_RANGE), sil_scores, "o-", color="darkorange")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score"); axes[1].set_title("Silhouette Score")
plt.tight_layout()
plt.savefig("kmeans_selection.png", dpi=150)
plt.show()
print("Saved kmeans_selection.png — inspect to choose K, then set K below.")

# --- 2b. Fit final K-Means with chosen K ------------------------------------
# Adjust K after inspecting the plots above.
K = 5

km_final = KMeans(n_clusters=K, random_state=SEED, n_init=50)
df["Cluster"] = km_final.fit_predict(X)

# Print cluster sizes and mean stats
print(f"\nK={K} cluster sizes:")
print(df["Cluster"].value_counts().sort_index())

cluster_means = df.groupby("Cluster")[STAT_COLS].mean()
print("\nCluster mean stats (raw):")
print(cluster_means.round(1).to_string())

# Label clusters by dominant stat profile (assigned after inspection)
# Edit these once you've seen the radar charts.
CLUSTER_NAMES = {
    0: "HP Tank",
    1: "Below Average",
    2: "All-Rounder",
    3: "Sp. Fighter",
    4: "Physical Wall",
}
df["Archetype"] = df["Cluster"].map(CLUSTER_NAMES)

# =============================================================================
# 3. SUPERVISED — KNN + DECISION TREE
# =============================================================================

from sklearn.model_selection import train_test_split

# K-Means labels come from the full dataset — split only for supervised eval
y = df["Cluster"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\nSupervised split: {len(X_train)} train / {len(X_test)} test")

# --- 3a. KNN -----------------------------------------------------------------
print("\n── KNN ──")
knn_scores = {}
for n in [3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=n)
    cv = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
    knn_scores[n] = cv.mean()
    print(f"  k={n}: CV {cv.mean():.3f} ± {cv.std():.3f}")

best_k = max(knn_scores, key=knn_scores.get)
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
knn_test_acc = knn_final.score(X_test, y_test)
print(f"  Best k={best_k} — held-out test accuracy: {knn_test_acc:.3f}")

# --- 3b. Decision Tree -------------------------------------------------------
print("\n── Decision Tree ──")
dt_scores = {}
for depth in [3, 4, 5, 6]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
    cv = cross_val_score(dt, X_train, y_train, cv=5, scoring="accuracy")
    dt_scores[depth] = cv.mean()
    print(f"  depth={depth}: CV {cv.mean():.3f} ± {cv.std():.3f}")

dt_final = DecisionTreeClassifier(max_depth=3, random_state=SEED)
dt_final.fit(X_train, y_train)
dt_test_acc = dt_final.score(X_test, y_test)
print(f"  Depth=3 — held-out test accuracy: {dt_test_acc:.3f}")

# Convert z-score thresholds back to raw stat values for readability
def raw_rules(tree, feature_names, scaler):
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != -2 else "leaf" for i in tree_.feature]
    lines = []

    def recurse(node, depth):
        indent = "|   " * depth
        if tree_.feature[node] != -2:
            fname = feature_name[node]
            fidx = feature_names.index(fname)
            threshold_raw = tree_.threshold[node] * scaler.scale_[fidx] + scaler.mean_[fidx]
            lines.append(f"{indent}|--- {fname} <= {threshold_raw:.1f}")
            recurse(tree_.children_left[node], depth + 1)
            lines.append(f"{indent}|--- {fname} > {threshold_raw:.1f}")
            recurse(tree_.children_right[node], depth + 1)
        else:
            counts = tree_.value[node][0]
            pred = int(np.argmax(counts))
            lines.append(f"{indent}|--- class: {CLUSTER_NAMES[pred]}")

    recurse(0, 0)
    return "\n".join(lines)

print("\nDecision rules (depth=3, raw stat values):")
print(raw_rules(dt_final, STAT_COLS, scaler))

# --- 3c. Persist models ------------------------------------------------------
import joblib
joblib.dump(knn_final, "knn_model.pkl")
joblib.dump(scaler,    "scaler.pkl")
print("\nSaved knn_model.pkl and scaler.pkl")
print("To classify a new Pokemon:")
print("  knn = joblib.load('knn_model.pkl')")
print("  scaler = joblib.load('scaler.pkl')")
print("  cluster = knn.predict(scaler.transform([[hp, atk, def_, sp_atk, sp_def, spd]]))")

# =============================================================================
# 4. VISUALISATION
# =============================================================================

COLORS = plt.cm.tab10.colors  # up to 10 clusters

# --- 4a. PCA Biplot ----------------------------------------------------------
pca = PCA(n_components=2, random_state=SEED)
coords = pca.fit_transform(X)
df["PC1"] = coords[:, 0]
df["PC2"] = coords[:, 1]

fig, ax = plt.subplots(figsize=(11, 8))
for c in range(K):
    mask = df["Cluster"] == c
    ax.scatter(
        df.loc[mask, "PC1"], df.loc[mask, "PC2"],
        color=COLORS[c], alpha=0.6, s=40, label=CLUSTER_NAMES[c]
    )

# Annotate a few notable Pokemon
NOTABLE = ["Mewtwo", "Shedinja", "Blissey", "Shuckle", "Ninjask", "Slaking",
           "Dragapult", "Garchomp", "Chansey"]
for _, row in df[df["Name"].isin(NOTABLE)].iterrows():
    ax.annotate(row["Name"], (row["PC1"], row["PC2"]),
                fontsize=7, alpha=0.9,
                xytext=(4, 4), textcoords="offset points")

# Stat loading arrows
loadings = pca.components_.T  # shape (6, 2)
scale = 3.5  # scale arrows to be visible
for i, stat in enumerate(STAT_COLS):
    ax.annotate("", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(loadings[i, 0] * scale * 1.12, loadings[i, 1] * scale * 1.12,
            stat, fontsize=8, ha="center", va="center", fontweight="bold")

var1, var2 = pca.explained_variance_ratio_ * 100
ax.set_xlabel(f"PC1 ({var1:.1f}% var)")
ax.set_ylabel(f"PC2 ({var2:.1f}% var)")
ax.set_title("Fully-Evolved Pokémon — PCA Biplot")
ax.legend(loc="upper left", fontsize=8)
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.axvline(0, color="grey", lw=0.5, ls="--")
plt.tight_layout()
plt.savefig("pca_biplot.png", dpi=150)
plt.show()
print("Saved pca_biplot.png")

# --- 4b. Radar Charts per Cluster (polished hexagonal grid) ------------------

BG_DARK   = "#12122a"   # figure background
BG_CELL   = "#1a1a3a"   # axes background
BAND_A    = "#22224a"   # alternating band fill (outer)
BAND_B    = "#1a1a3a"   # alternating band fill (inner)
GRID_COL  = "#ffffff"   # grid / spoke colour

# Per-archetype colours chosen for contrast on dark background
ARCHETYPE_COLORS = {
    "HP Tank":       "#4FC3F7",  # sky blue
    "Below Average": "#90A4AE",  # steel grey
    "All-Rounder":   "#81C784",  # soft green
    "Sp. Fighter":   "#CE93D8",  # lavender
    "Physical Wall": "#FFB74D",  # warm amber
}

def radar_chart(ax, values, labels, color, title, n_poke):
    from matplotlib.patches import Polygon as MplPolygon

    N = len(labels)
    angles = [math.pi / 2 - n / N * 2 * math.pi for n in range(N)]
    vx = np.array([math.cos(a) for a in angles])
    vy = np.array([math.sin(a) for a in angles])
    max_val = 160.0

    ax.set_facecolor(BG_CELL)

    # Alternating filled bands (outermost first so inner bands paint over)
    for level, band_col in zip([160, 120, 80, 40], [BAND_A, BAND_B, BAND_A, BAND_B]):
        s = level / max_val
        pts = list(zip(vx * s, vy * s))
        poly = MplPolygon(pts, closed=True, facecolor=band_col,
                          edgecolor=GRID_COL, linewidth=0.6,
                          alpha=0.6, zorder=1)
        ax.add_patch(poly)

    # Level labels (beside the rightmost spoke)
    for level in [40, 80, 120]:
        s = level / max_val
        ax.text(vx[1] * s + 0.04, vy[1] * s, str(level),
                fontsize=6, color=GRID_COL, alpha=0.5,
                ha="left", va="center", zorder=2)

    # Spokes
    for x, y in zip(vx, vy):
        ax.plot([0, x], [0, y], color=GRID_COL, lw=0.6, alpha=0.25, zorder=2)

    # Stat polygon
    scaled = np.array(values) / max_val
    px = np.append(vx * scaled, vx[0] * scaled[0])
    py = np.append(vy * scaled, vy[0] * scaled[0])
    ax.fill(px, py, color=color, alpha=0.45, zorder=3)
    ax.plot(px, py, color=color, lw=2.5, zorder=4)

    # Vertex dots
    ax.scatter(vx * scaled, vy * scaled,
               color=color, s=35, zorder=5, edgecolors="white", linewidths=0.5)

    # Stat value labels — nudged outward from each vertex
    for i, val in enumerate(values):
        nudge = 0.14
        ax.text(vx[i] * scaled[i] + vx[i] * nudge,
                vy[i] * scaled[i] + vy[i] * nudge,
                str(int(round(val))),
                fontsize=7.5, color="white", fontweight="bold",
                ha="center", va="center", zorder=6)

    # Stat name labels at outer ring
    label_r = 1.26
    for i, label in enumerate(labels):
        ax.text(vx[i] * label_r, vy[i] * label_r, label,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", zorder=6)

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title}", size=12, pad=16,
                 color="white", fontweight="bold")
    ax.text(0, -1.52, f"n = {n_poke}", ha="center", va="top",
            fontsize=8, color=color, alpha=0.85, zorder=6)

ncols = 3
nrows = math.ceil(K / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
fig.patch.set_facecolor(BG_DARK)
axes_flat = axes.flatten() if K > 1 else [axes]

for c in range(K):
    mask = df["Cluster"] == c
    means = df.loc[mask, STAT_COLS].mean().values
    name = CLUSTER_NAMES[c]
    color = ARCHETYPE_COLORS[name]
    radar_chart(axes_flat[c], means, STAT_LABELS, color, name, mask.sum())

for i in range(K, len(axes_flat)):
    axes_flat[i].set_visible(False)

fig.suptitle("Pokémon Archetypes — Mean Base Stats",
             fontsize=15, color="white", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("radar_clusters.png", dpi=150, bbox_inches="tight",
            facecolor=BG_DARK)
plt.show()
print("Saved radar_clusters.png")

# --- 4b-ii. Radar Charts — Pokémon light style -------------------------------

import matplotlib.colors as mcolors

def darken(hex_color, factor=0.55):
    """Return a darkened version of hex_color (factor < 1 = darker)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)

def tint(hex_color, factor=0.15):
    """Return a very light tint of hex_color for cell background."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r + (1 - r) * (1 - factor), g + (1 - g) * (1 - factor), b + (1 - b) * (1 - factor))

def radar_chart_light(ax, values, labels, color, title, n_poke):
    from matplotlib.patches import Polygon as MplPolygon

    N = len(labels)
    angles = [math.pi / 2 - n / N * 2 * math.pi for n in range(N)]
    vx = np.array([math.cos(a) for a in angles])
    vy = np.array([math.sin(a) for a in angles])
    max_val = 160.0
    dark_color = darken(color, 0.6)
    cell_bg    = tint(color, 0.12)

    ax.set_facecolor(cell_bg)

    # Outer hexagon filled with light archetype colour accent
    outer_fill = MplPolygon(list(zip(vx, vy)), closed=True,
                            facecolor=color, alpha=0.18,
                            edgecolor="none", zorder=0)
    ax.add_patch(outer_fill)

    # Inner grid hexagons — light grey
    for level in [40, 80, 120]:
        s = level / max_val
        pts = list(zip(vx * s, vy * s))
        poly = MplPolygon(pts, closed=True, facecolor="none",
                          edgecolor="#cccccc", linewidth=0.8, zorder=1)
        ax.add_patch(poly)

    # Thick outer border in archetype colour
    outer_pts = list(zip(vx, vy))
    outer_border = MplPolygon(outer_pts, closed=True, facecolor="none",
                              edgecolor=color, linewidth=3.5, zorder=5)
    ax.add_patch(outer_border)

    # Spokes
    for x, y in zip(vx, vy):
        ax.plot([0, x], [0, y], color="#cccccc", lw=0.8, zorder=2)

    # Level labels
    for level in [40, 80, 120]:
        s = level / max_val
        ax.text(vx[1] * s + 0.04, vy[1] * s, str(level),
                fontsize=6, color="#999999", ha="left", va="center", zorder=2)

    # Stat polygon — dark fill
    scaled = np.array(values) / max_val
    px = np.append(vx * scaled, vx[0] * scaled[0])
    py = np.append(vy * scaled, vy[0] * scaled[0])
    ax.fill(px, py, color=color, alpha=0.35, zorder=3)
    ax.plot(px, py, color=color, lw=2.5, zorder=4)

    # Vertex dots
    ax.scatter(vx * scaled, vy * scaled,
               color=color, s=35, zorder=6, edgecolors="white", linewidths=0.8)

    # Stat value labels
    for i, val in enumerate(values):
        nudge = 0.15
        ax.text(vx[i] * scaled[i] + vx[i] * nudge,
                vy[i] * scaled[i] + vy[i] * nudge,
                str(int(round(val))),
                fontsize=7.5, color="#555555", fontweight="bold",
                ha="center", va="center", zorder=7)

    # Stat name labels
    label_r = 1.26
    for i, label in enumerate(labels):
        ax.text(vx[i] * label_r, vy[i] * label_r, label,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="#222222", zorder=6)

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, size=12, pad=16, color="#222222", fontweight="bold")
    ax.text(0, -1.52, f"n = {n_poke}", ha="center", va="top",
            fontsize=8, color=dark_color, alpha=0.9, zorder=6)

# Use a 6-column GridSpec: top row spans cols 0-1, 2-3, 4-5
# bottom row spans cols 1-2, 3-4 — centred under the gaps
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("white")
gs = GridSpec(2, 6, figure=fig, hspace=0.1, wspace=0.3)

top_slices    = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6]]
bottom_slices = [gs[1, 1:3], gs[1, 3:5]]
grid_slices   = top_slices + bottom_slices

for c in range(K):
    ax = fig.add_subplot(grid_slices[c])
    mask = df["Cluster"] == c
    means = df.loc[mask, STAT_COLS].mean().values
    name = CLUSTER_NAMES[c]
    color = ARCHETYPE_COLORS[name]
    radar_chart_light(ax, means, STAT_LABELS, color, name, mask.sum())

fig.suptitle("Pokémon Archetypes — Mean Base Stats",
             fontsize=15, color="#222222", fontweight="bold", y=1.02)
plt.savefig("radar_clusters_light.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
print("Saved radar_clusters_light.png")

# --- 4c. Type Heatmap --------------------------------------------------------
# Each Pokemon may have two types (e.g. "Fire  Flying") — split and explode
type_df = df[["Archetype", "Type"]].copy()
type_df["Type"] = type_df["Type"].str.strip().str.split(r"\s+")
type_df = type_df.explode("Type")

type_counts = (
    type_df.groupby(["Type", "Archetype"])
    .size()
    .unstack(fill_value=0)
)
# Normalise by archetype total so columns are comparable
type_pct = type_counts.div(type_counts.sum(axis=0), axis=1) * 100

archetype_order = [CLUSTER_NAMES[i] for i in range(K)]
type_pct = type_pct.reindex(columns=archetype_order)
type_pct = type_pct.loc[type_pct.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(12, 11))
im = ax.imshow(type_pct.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(K))
ax.set_xticklabels(archetype_order, fontsize=12, fontweight="bold")
ax.set_yticks(range(len(type_pct)))
ax.set_yticklabels(type_pct.index, fontsize=10)
ax.tick_params(axis="x", pad=8)
cbar = plt.colorbar(im, ax=ax, label="% of archetype", shrink=0.8)
cbar.ax.tick_params(labelsize=9)
cbar.set_label("% of archetype", fontsize=10)
ax.set_title("Type Distribution by Archetype (% of archetype total)",
             fontsize=13, pad=14)
vmax = type_pct.values.max()
for i in range(len(type_pct)):
    for j in range(len(archetype_order)):
        val = type_pct.values[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                fontsize=8.5, fontweight="bold",
                color="white" if val > vmax * 0.55 else "#333333")
plt.tight_layout()
plt.savefig("type_heatmap.png", dpi=150)
plt.show()
print("Saved type_heatmap.png")

# --- 4d. Generation Breakdown ------------------------------------------------
gen_counts = (
    df.groupby(["Generation", "Archetype"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=archetype_order)
)

fig, ax = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(gen_counts))
for i, archetype in enumerate(archetype_order):
    vals = gen_counts[archetype].values
    ax.bar(gen_counts.index, vals, bottom=bottom,
           color=COLORS[i], label=archetype, edgecolor="white", linewidth=0.5)
    bottom += vals

ax.set_xlabel("Generation")
ax.set_ylabel("Number of Pokémon")
ax.set_title("Archetype Distribution by Generation")
ax.set_xticks(gen_counts.index)
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("generation_breakdown.png", dpi=150)
plt.show()
print("Saved generation_breakdown.png")

# =============================================================================
# 5. EXPORT — labelled Pokedex
# =============================================================================

out_cols = ["Dex_Number", "Name", "Type", "Generation"] + STAT_COLS + ["Total", "Cluster", "Archetype"]
df[out_cols].to_csv("pokedex_clustered.csv", index=False)
print("\nSaved pokedex_clustered.csv")
print(df.groupby("Archetype")["Name"].apply(lambda x: ", ".join(x.sample(min(5, len(x)), random_state=SEED))).to_string())
