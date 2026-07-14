"""
SCRIPT 06 - Film Embeddings, UMAP + HDBSCAN Clustering
=======================================================
Transforms each film into a numeric vector, compresses to 2D
with UMAP, and finds taste clusters with HDBSCAN.
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'processed', 'movies_enriched.csv')
OUT_DIR = os.path.join(ROOT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor':  '#0a0a0f',
    'axes.facecolor':    '#111118',
    'axes.edgecolor':    '#333344',
    'axes.labelcolor':   '#9999b8',
    'axes.titlecolor':   '#e8e8f0',
    'axes.titlesize':    13,
    'axes.titlepad':     14,
    'axes.grid':         True,
    'grid.color':        '#1e1e2e',
    'grid.linewidth':    0.5,
    'xtick.color':       '#666680',
    'ytick.color':       '#666680',
    'text.color':        '#e8e8f0',
    'font.family':       'monospace',
})
GOLD   = '#e8c96a'
GREEN  = '#00c030'
RED    = '#e84040'
TEAL   = '#00ccaa'
PURPLE = '#9966ff'

# ─────────────────────────────────────────────────────────────
# STEP 1 - BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_FILE)
df = df[df['Rating'].notna() & df['genres'].notna()].copy()
df = df.reset_index(drop=True)

# Numeric features — normalised to 0-1 range
df['year_norm']    = (df['Year'] - df['Year'].min()) / (df['Year'].max() - df['Year'].min())
df['decade_norm']  = ((df['Year'] // 10 * 10) - 1920) / 100
df['rating_norm']  = df['Rating'] / 5.0
df['runtime_norm'] = df['runtime'].fillna(df['runtime'].median()) / 240
df['tmdb_norm']    = df['tmdb_rating'].fillna(0) / 10.0
numeric_features   = ['rating_norm', 'year_norm', 'decade_norm', 'runtime_norm', 'tmdb_norm']

# Genre binary flags — weighted 1.5x
top_genres = df['genres'].str.split('|').explode().value_counts().head(20).index.tolist()
for genre in top_genres:
    col = f'genre_{genre.lower().replace(" ", "_")}'
    df[col] = df['genres'].str.contains(genre, na=False).astype(float) * 1.5
genre_features = [f'genre_{g.lower().replace(" ", "_")}' for g in top_genres]

# Country binary flags
top_countries = df['countries'].dropna().str.split('|').explode().value_counts().head(10).index.tolist()
for country in top_countries:
    df[f'country_{country.lower()}'] = df['countries'].fillna('').str.contains(country).astype(float)
country_features = [f'country_{c.lower()}' for c in top_countries]

# Director binary flags — weighted 2x (very informative)
top_dirs = df['directors'].dropna().str.split('|').explode().value_counts().head(30).index.tolist()
for director in top_dirs:
    col = f'dir_{director.lower().replace(" ", "_").replace(".", "")}'
    df[col] = df['directors'].fillna('').str.contains(director, regex=False).astype(float) * 2.0
dir_features = [f'dir_{d.lower().replace(" ", "_").replace(".", "")}' for d in top_dirs]

all_features = numeric_features + genre_features + country_features + dir_features
X = df[all_features].fillna(0).values

print(f"  {X.shape[0]} films x {X.shape[1]} features")

# ─────────────────────────────────────────────────────────────
# STEP 2 - UMAP: 65 dimensions -> 2
# ─────────────────────────────────────────────────────────────

import umap

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42
)
embedding = reducer.fit_transform(X)

df['umap_x'] = embedding[:, 0]
df['umap_y'] = embedding[:, 1]
print(f"  Done. {X.shape} -> {embedding.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 3 - HDBSCAN clustering
# ─────────────────────────────────────────────────────────────

import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
labels = clusterer.fit_predict(embedding)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
df['cluster'] = labels

print(f"  {n_clusters} clusters found, {n_noise} noise points")
for cid in sorted(set(labels)):
    if cid == -1:
        continue
    mask = df['cluster'] == cid
    top_g = df[mask]['genres'].str.split('|').explode().value_counts().head(2).index.tolist()
    print(f"  Cluster {cid:2d}: {mask.sum():3d} films | "
          f"avg {df[mask]['Rating'].mean():.2f} | "
          f"~{df[mask]['Year'].mean():.0f} | "
          f"{', '.join(top_g)}")

# ─────────────────────────────────────────────────────────────
# STEP 4 - VISUALISE
# ─────────────────────────────────────────────────────────────
cluster_ids = sorted(set(labels))
cmap_c = plt.cm.tab20
color_map = {c: cmap_c(i / max(len(cluster_ids), 1)) for i, c in enumerate(cluster_ids)}
color_map[-1] = (0.2, 0.2, 0.25, 0.25)

# CHART 1: coloured by cluster
fig, ax = plt.subplots(figsize=(15, 11))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')
fig.suptitle(
    f'My Film Galaxy  //  {n_clusters} taste clusters (HDBSCAN)  //  '
    f'position = UMAP of genres + rating + director',
    color=GOLD, fontsize=12
)

noise_mask = df['cluster'] == -1
ax.scatter(df[noise_mask]['umap_x'], df[noise_mask]['umap_y'],
           c='#2a2a3a', s=10, alpha=0.25, zorder=1)

for cid in [c for c in cluster_ids if c != -1]:
    mask = df['cluster'] == cid
    cdf  = df[mask]
    ax.scatter(cdf['umap_x'], cdf['umap_y'],
               c=[color_map[cid]], s=cdf['Rating'].fillna(2.5) * 8,
               alpha=0.75, zorder=2, edgecolors='none')
    cx = cdf['umap_x'].mean()
    cy = cdf['umap_y'].mean()
    top_g = cdf['genres'].str.split('|').explode().value_counts().head(2).index.tolist()
    ax.text(cx, cy, f"#{cid}\n{' · '.join(top_g)}\n{len(cdf)} films",
            ha='center', va='center', fontsize=7.5, color='#eeeeee', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a14',
                      edgecolor=color_map[cid][:3] + (0.6,), alpha=0.85, linewidth=1.2))

# Label a sample of 5-star films
for _, row in df[df['Rating'] == 5.0].sample(min(15, (df['Rating'] == 5.0).sum()), random_state=42).iterrows():
    short = row['Name'][:20] + '...' if len(row['Name']) > 20 else row['Name']
    ax.text(row['umap_x'], row['umap_y'] + 0.15, short,
            ha='center', fontsize=6, color=GOLD, alpha=0.85,
            path_effects=[pe.withStroke(linewidth=2, foreground='#0a0a0f')])

ax.set_xlabel('UMAP dimension 1  (relative position matters, not absolute value)')
ax.set_ylabel('UMAP dimension 2')
ax.set_title('Each point = one film  //  size = my rating  //  gold = 5-star films')
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '20_film_galaxy_clusters.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/20_film_galaxy_clusters.png")

# CHART 2: coloured by rating
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')
fig.suptitle('My Film Galaxy  //  coloured by my rating', color=GOLD, fontsize=12)

sc = ax.scatter(df['umap_x'], df['umap_y'],
                c=df['Rating'], cmap=plt.cm.RdYlGn,
                norm=mcolors.Normalize(vmin=0.5, vmax=5.0),
                s=df['Rating'].fillna(2.5) * 8, alpha=0.75, edgecolors='none')

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('My rating', color='#9999b8')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#9999b8')

for _, row in df[df['Rating'] == 5.0].iterrows():
    short = row['Name'][:18] + '...' if len(row['Name']) > 18 else row['Name']
    ax.text(row['umap_x'], row['umap_y'] + 0.12, short,
            ha='center', fontsize=5.5, color=GOLD,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#0a0a0f')])

ax.set_xlabel('UMAP dimension 1')
ax.set_ylabel('UMAP dimension 2')
ax.set_title('Green = loved it  //  Red = hated it  //  gold labels = 5-star films')
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '21_film_galaxy_ratings.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/21_film_galaxy_ratings.png")

# CHART 3: coloured by decade
decade_colors = {
    1920: '#9b59b6', 1930: '#8e44ad', 1950: '#2471a3', 1960: '#1a5276',
    1970: '#16a085', 1980: '#1e8449', 1990: '#f39c12',
    2000: '#e67e22', 2010: GOLD,      2020: GREEN
}
df['decade'] = (df['Year'] // 10 * 10).astype(int)

fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')
fig.suptitle('My Film Galaxy  //  coloured by decade', color=GOLD, fontsize=12)

for decade, color in decade_colors.items():
    mask = df['decade'] == decade
    if mask.sum() == 0:
        continue
    ax.scatter(df[mask]['umap_x'], df[mask]['umap_y'],
               c=color, s=df[mask]['Rating'].fillna(2.5) * 7,
               alpha=0.75, edgecolors='none', label=f'{decade}s ({mask.sum()})')

ax.legend(loc='lower right', facecolor='#1a1a24', edgecolor='#333344',
          labelcolor='#ccccdd', fontsize=8, ncol=2)
ax.set_xlabel('UMAP dimension 1')
ax.set_ylabel('UMAP dimension 2')
ax.set_title('Size = my rating')
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '22_film_galaxy_decades.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/22_film_galaxy_decades.png")

# Save for script 07
df[['Name', 'Year', 'Rating', 'genres', 'directors',
    'cluster', 'umap_x', 'umap_y']].to_csv(
    os.path.join(ROOT_DIR, 'data', 'processed', 'embeddings.csv'),
    index=False, encoding='utf-8')
np.save(os.path.join(ROOT_DIR, 'data', 'processed', 'feature_matrix.npy'), X)

print()
print("=" * 50)
print("  3 charts saved in output/")
print("=" * 50)
print(f"  20 - Galaxy by cluster")
print(f"  21 - Galaxy by rating")
print(f"  22 - Galaxy by decade")
print(f"\n  {n_clusters} clusters | {n_noise} noise points")
print(f"  Embeddings saved for script 07.")
