"""
SCRIPT 04 - Advanced visualisations
=====================================
6 creative charts going deeper into patterns, evolution and networks.

Charts:
  09 - Heatmap: genres watched per year
  10 - Heatmap: genre combinations and how I rate them
  11 - Treemap: genre universe (size = films, colour = rating)
  12 - Rating evolution: am I getting more critical over time?
  13 - Director loyalty: consistency vs avg rating
  14 - Network: directors connected by shared actors
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

# PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, '..')
DATA_FILE  = os.path.join(ROOT_DIR, 'data', 'processed', 'movies_enriched.csv')
DIARY_FILE = os.path.join(ROOT_DIR, 'data', 'raw', 'diary.csv')
OUT_DIR    = os.path.join(ROOT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# STYLE
plt.rcParams.update({
    'figure.facecolor':  '#0a0a0f',
    'axes.facecolor':    '#111118',
    'axes.edgecolor':    '#333344',
    'axes.labelcolor':   '#9999b8',
    'axes.titlecolor':   '#e8e8f0',
    'axes.titlesize':    13,
    'axes.titlepad':     14,
    'axes.grid':         False,
    'xtick.color':       '#666680',
    'ytick.color':       '#666680',
    'text.color':        '#e8e8f0',
    'font.family':       'monospace',
})
GOLD   = '#e8c96a'
GREEN  = '#00c030'
RED    = '#e84040'
BLUE   = '#4488ff'
PURPLE = '#9966ff'
TEAL   = '#00ccaa'

# LOAD
df = pd.read_csv(DATA_FILE)
diary = pd.read_csv(DIARY_FILE)
diary['Date'] = pd.to_datetime(diary['Date'])
diary['year'] = diary['Date'].dt.year
diary['month'] = diary['Date'].dt.month
diary['yearmonth'] = diary['Date'].dt.to_period('M')

# Merge diary dates into main df
# Rename diary Date to watched_date to avoid clash with ratings Date column
diary_slim = diary[['Name','Year','Date','year','month','yearmonth']].copy()
diary_slim = diary_slim.rename(columns={'Date': 'watched_date'})
df2 = df.merge(diary_slim, on=['Name','Year'], how='left')

# Exploded versions
genres_exp    = df[df['genres'].notna()].assign(genre=df['genres'].str.split('|')).explode('genre')
directors_exp = df[df['directors'].notna()].assign(director=df['directors'].str.split('|')).explode('director')
cast_exp      = df[df['cast'].notna()].assign(actor=df['cast'].str.split('|')).explode('actor')
genres_diary  = df2[df2['genres'].notna()].assign(genre=df2['genres'].str.split('|')).explode('genre')
# Note: df2 now has watched_date instead of Date from diary

print(f"  {len(df)} films ready.\n")

# ─────────────────────────────────────────────────────────────
# CHART 1 - HEATMAP: genres as % of each year's viewing
# Using % instead of raw counts so 2025 doesn't dominate visually
# ─────────────────────────────────────────────────────────────

pivot_raw = genres_diary.groupby(['year','genre']).size().unstack(fill_value=0)
pivot_raw = pivot_raw[pivot_raw.index.isin([2024, 2025, 2026])]
top_genres = genres_exp.groupby('genre').size().nlargest(12).index
pivot_raw = pivot_raw[top_genres]

# % of each year's total — makes years comparable regardless of volume
pivot_pct = pivot_raw.div(pivot_raw.sum(axis=1), axis=0) * 100
pivot_pct = pivot_pct.T  # genres as rows, years as columns

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle(
    "Genre Mix Per Year  //  % of that year's total viewing",
    fontsize=14, color=GOLD, y=1.01
)

im = ax.imshow(pivot_pct.values, aspect='auto', cmap='YlOrRd', vmin=0, vmax=32)

ax.set_xticks(range(len(pivot_pct.columns)))
ax.set_xticklabels([str(int(y)) for y in pivot_pct.columns], fontsize=11)
ax.set_yticks(range(len(pivot_pct.index)))
ax.set_yticklabels(pivot_pct.index, fontsize=10)
ax.set_facecolor('#111118')

# Each cell shows % and raw count
for i, genre in enumerate(pivot_pct.index):
    for j, year in enumerate(pivot_pct.columns):
        pct_val = pivot_pct.values[i, j]
        raw_val = pivot_raw.loc[year, genre] if year in pivot_raw.index and genre in pivot_raw.columns else 0
        rgba = plt.cm.YlOrRd(pct_val / 32)
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        color = '#111118' if luminance > 0.5 else '#ffffff'
        txt = ax.text(j, i, f'{pct_val:.0f}%\n({int(raw_val)})',
                ha='center', va='center',
                fontsize=8.5, color=color, fontweight='bold', linespacing=1.4)
        txt.set_path_effects([
            __import__('matplotlib.patheffects', fromlist=['withStroke']).withStroke(linewidth=2, foreground='#000000' if color == '#ffffff' else '#ffffff')
        ])

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("% of year's viewing", color='#9999b8')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#9999b8')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '09_genre_heatmap_year.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/09_genre_heatmap_year.png")

# ─────────────────────────────────────────────────────────────
# CHART 2 - HEATMAP: genre combinations and avg rating
# ─────────────────────────────────────────────────────────────

from itertools import combinations

top_g = genres_exp.groupby('genre').size().nlargest(10).index.tolist()

matrix = pd.DataFrame(index=top_g, columns=top_g, dtype=float)
counts = pd.DataFrame(index=top_g, columns=top_g, dtype=int)

for g1 in top_g:
    for g2 in top_g:
        if g1 == g2:
            mask = df['genres'].str.contains(g1, na=False)
        else:
            mask = (df['genres'].str.contains(g1, na=False) &
                    df['genres'].str.contains(g2, na=False))
        vals = df[mask]['Rating']
        matrix.loc[g1, g2] = vals.mean() if len(vals) >= 3 else np.nan
        counts.loc[g1, g2] = len(vals)

matrix = matrix.astype(float)

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('My Avg Rating by Genre Combination', fontsize=15, color=GOLD, y=1.01)

cmap_combo = plt.cm.RdYlGn
im = ax.imshow(matrix.values, cmap=cmap_combo, aspect='auto', vmin=2.0, vmax=5.0)

ax.set_xticks(range(len(top_g)))
ax.set_xticklabels(top_g, rotation=35, ha='right', fontsize=9)
ax.set_yticks(range(len(top_g)))
ax.set_yticklabels(top_g, fontsize=9)
ax.set_facecolor('#111118')

for i in range(len(top_g)):
    for j in range(len(top_g)):
        val = matrix.values[i, j]
        if not np.isnan(val):
            brightness = (val - 2.0) / 3.0
            color = '#111118' if brightness > 0.35 else '#eeeeee'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

plt.colorbar(im, ax=ax, label='Avg rating').ax.yaxis.label.set_color('#9999b8')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '10_genre_combo_heatmap.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/10_genre_combo_heatmap.png")

# ─────────────────────────────────────────────────────────────
# CHART 3 - TREEMAP: genre universe — cleaner layout
# ─────────────────────────────────────────────────────────────

import squarify

genre_stats = genres_exp.groupby('genre').agg(
    films=('Name', 'count'),
    avg_rating=('Rating', 'mean')
).query('films >= 8').sort_values('films', ascending=False)

norm_tree  = mcolors.Normalize(vmin=2.5, vmax=4.5)
cmap_tree  = plt.cm.RdYlGn
colors     = [cmap_tree(norm_tree(r)) for r in genre_stats['avg_rating']]

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')
fig.suptitle(
    "My Genre Universe  //  size = films watched  //  colour = avg rating  (green = love it, red = don't)",
    fontsize=13, color=GOLD
)

# Labels adapt to tile size: large tiles show more info
labels = []
for genre, row in genre_stats.iterrows():
    if row['films'] >= 100:
        labels.append(f"{genre}\n{int(row['films'])} films\navg {row['avg_rating']:.2f}")
    elif row['films'] >= 40:
        labels.append(f"{genre}\n{int(row['films'])} · {row['avg_rating']:.2f}")
    else:
        labels.append(f"{genre}\n{int(row['films'])}")

squarify.plot(
    sizes=genre_stats['films'],
    label=labels,
    color=colors,
    alpha=0.9,
    ax=ax,
    pad=True,
    text_kwargs={
        'fontsize': 10,
        'color': '#111118',
        'fontfamily': 'monospace',
        'fontweight': 'bold',
        'linespacing': 1.5,
    }
)

ax.set_axis_off()

sm_tree = plt.cm.ScalarMappable(cmap=cmap_tree, norm=norm_tree)
sm_tree.set_array([])
cbar = plt.colorbar(sm_tree, ax=ax, orientation='horizontal', fraction=0.025, pad=0.01)
cbar.set_label('Avg rating', color='#9999b8', fontsize=10)
plt.setp(cbar.ax.xaxis.get_ticklabels(), color='#9999b8')
cbar.ax.xaxis.set_tick_params(color='#9999b8')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '11_genre_treemap.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/11_genre_treemap.png")

# CHART 4 - RATING EVOLUTION over time

df2_sorted = df2[df2['watched_date'].notna()].sort_values('watched_date').copy()
df2_sorted['rolling_avg'] = df2_sorted['Rating'].rolling(window=30, min_periods=10).mean()
df2_sorted['yearmonth_str'] = df2_sorted['yearmonth'].astype(str)

monthly_avg = df2_sorted.groupby('yearmonth_str').agg(
    avg_rating=('Rating', 'mean'),
    count=('Rating', 'count')
).reset_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), height_ratios=[3, 1])
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('Am I Getting More Critical Over Time?', fontsize=15, color=GOLD)

# Rolling average line
x = range(len(df2_sorted))
ax1.scatter(x, df2_sorted['Rating'], alpha=0.08, s=8, color=TEAL, zorder=2)
ax1.plot(x, df2_sorted['rolling_avg'], color=GOLD, linewidth=2.5, zorder=3,
         label='Rolling avg (30 films)')
ax1.axhline(y=df['Rating'].mean(), color='#555566', linestyle='--',
            linewidth=1, alpha=0.7, label=f'Overall avg ({df["Rating"].mean():.2f})')

# Shade above/below overall avg
avg = df['Rating'].mean()
ax1.fill_between(x, df2_sorted['rolling_avg'], avg,
                 where=df2_sorted['rolling_avg'] >= avg,
                 alpha=0.15, color=GREEN, interpolate=True)
ax1.fill_between(x, df2_sorted['rolling_avg'], avg,
                 where=df2_sorted['rolling_avg'] < avg,
                 alpha=0.15, color=RED, interpolate=True)

ax1.set_ylabel('Rating')
ax1.set_ylim(0.5, 5.5)
ax1.set_facecolor('#111118')
ax1.legend(facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8')
ax1.set_xticks([])

# Monthly film count bar
ax2.bar(range(len(monthly_avg)), monthly_avg['count'],
        color=PURPLE, alpha=0.7, zorder=3)
ax2.set_xticks(range(len(monthly_avg)))
ax2.set_xticklabels(monthly_avg['yearmonth_str'], rotation=45, ha='right', fontsize=7)
ax2.set_ylabel('Films/month')
ax2.set_facecolor('#111118')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '12_rating_evolution.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/12_rating_evolution.png")


# ─────────────────────────────────────────────────────────────
# CHART 5 - DIRECTOR LOYALTY: consistency vs avg rating
# Fixed: quadrant labels placed inside axes using transforms
# Bigger figure + more separation between nodes and labels
# ─────────────────────────────────────────────────────────────

from adjustText import adjust_text

dir_stats = directors_exp.groupby('director').agg(
    films=('Name', 'count'),
    avg=('Rating', 'mean'),
    std=('Rating', 'std')
).query('films >= 4').fillna(0)

# Bigger figure so nodes and labels have more breathing room
fig, ax = plt.subplots(figsize=(16, 11))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle(
    'Director Loyalty  //  x = consistency (low = always consistent)  //  y = how much I like them',
    fontsize=12, color=GOLD
)

scatter = ax.scatter(
    dir_stats['std'],
    dir_stats['avg'],
    s=dir_stats['films'] * 100,
    c=dir_stats['avg'],
    cmap='RdYlGn',
    vmin=2.0, vmax=5.0,
    alpha=0.85,
    zorder=3,
    edgecolors='#333344',
    linewidths=0.5
)

texts = []
for director, row in dir_stats.iterrows():
    texts.append(ax.text(
        row['std'], row['avg'], director,
        fontsize=8.5, color='#ddddee',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#1a1a2e', edgecolor='none', alpha=0.75)
    ))

adjust_text(
    texts, ax=ax,
    arrowprops=dict(arrowstyle='-', color='#555577', lw=0.5),
    expand=(2.5, 3.0),
    force_text=(1.5, 2.0),
    force_points=(2.0, 2.5),
)

# Dividing lines
avg_rating = df['Rating'].mean()
med_std = dir_stats['std'].median()
ax.axhline(y=avg_rating, color='#444455', linestyle='--', linewidth=0.8)
ax.axvline(x=med_std,    color='#444455', linestyle='--', linewidth=0.8)

# Quadrant labels using axis transform so they stay inside the plot
# (0,0) = bottom-left corner of axes, (1,1) = top-right
ax.text(0.02, 0.97, 'Love consistently',   transform=ax.transAxes,
        fontsize=9, color=GREEN,    alpha=0.8, va='top')
ax.text(0.52, 0.97, 'Love but unpredictable', transform=ax.transAxes,
        fontsize=9, color=GOLD,     alpha=0.8, va='top')
ax.text(0.02, 0.03, 'Consistently meh',    transform=ax.transAxes,
        fontsize=9, color='#666680', alpha=0.8, va='bottom')
ax.text(0.52, 0.03, 'Wildly inconsistent', transform=ax.transAxes,
        fontsize=9, color=RED,      alpha=0.8, va='bottom')

ax.set_xlabel('Rating std dev  (0 = always rate the same,  2 = very inconsistent)')
ax.set_ylabel('Avg rating I give')
ax.set_facecolor('#111118')
plt.colorbar(scatter, ax=ax, label='Avg rating').ax.yaxis.label.set_color('#9999b8')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '13_director_loyalty.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/13_director_loyalty.png")

# CHART 6 - NETWORK: directors connected by shared actors

import networkx as nx

# For each film, get director + cast
# Two directors are connected if they share an actor in my watched films
film_directors = df[df['directors'].notna() & df['cast'].notna()].copy()
film_directors['dir_list']  = film_directors['directors'].str.split('|')
film_directors['cast_list'] = film_directors['cast'].str.split('|')

# Build actor -> directors mapping
actor_to_dirs = {}
for _, row in film_directors.iterrows():
    for actor in row['cast_list']:
        for director in row['dir_list']:
            actor_to_dirs.setdefault(actor, set()).add(director)

# Only keep directors with >= 3 films
top_dirs = set(directors_exp.groupby('director').size()[
    directors_exp.groupby('director').size() >= 3
].index)

# Build graph
G = nx.Graph()
for actor, dirs in actor_to_dirs.items():
    dirs = [d for d in dirs if d in top_dirs]
    if len(dirs) >= 2:
        for d1, d2 in combinations(sorted(dirs), 2):
            if G.has_edge(d1, d2):
                G[d1][d2]['weight'] += 1
            else:
                G.add_edge(d1, d2, weight=1)

# Keep only edges with weight >= 2 (shared 2+ actors)
edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 2]
G.remove_edges_from(edges_to_remove)
# Remove isolated nodes
G.remove_nodes_from(list(nx.isolates(G)))

print(f"  Network: {G.number_of_nodes()} directors, {G.number_of_edges()} connections")

# Node attributes: avg rating and film count
dir_avg = directors_exp.groupby('director')['Rating'].mean()
dir_count = directors_exp.groupby('director').size()
for node in G.nodes():
    G.nodes[node]['avg_rating'] = dir_avg.get(node, 3.0)
    G.nodes[node]['films'] = dir_count.get(node, 1)

fig, ax = plt.subplots(figsize=(15, 11))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle(
    'Director Network  //  connected by shared actors  //  colour = my avg rating  //  size = films watched',
    fontsize=12, color=GOLD
)
ax.set_facecolor('#0a0a0f')

pos = nx.spring_layout(G, k=2.5, iterations=60, seed=42)

# Edge width by shared actor count
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
max_w = max(edge_weights) if edge_weights else 1

nx.draw_networkx_edges(G, pos, ax=ax,
    width=[0.5 + 2.0 * (w / max_w) for w in edge_weights],
    edge_color='#2a2a4a', alpha=0.7)

# Node colour by avg rating
norm = mcolors.Normalize(vmin=2.5, vmax=5.0)
cmap = plt.cm.RdYlGn
node_colors = [cmap(norm(G.nodes[n]['avg_rating'])) for n in G.nodes()]
node_sizes = [G.nodes[n]['films'] * 120 for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, ax=ax,
    node_color=node_colors, node_size=node_sizes, alpha=0.9)

nx.draw_networkx_labels(G, pos, ax=ax,
    font_size=7, font_color='#ddddee',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#0a0a14', edgecolor='none', alpha=0.6))

ax.set_axis_off()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
cbar.set_label('Avg rating I give', color='#9999b8')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#9999b8')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '14_director_network.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/14_director_network.png")

# DONE
print()
print("=" * 50)
print("  DONE! 6 charts saved in output/")
print("=" * 50)
print("  09 - Genre heatmap by year")
print("  10 - Genre combination heatmap")
print("  11 - Genre treemap")
print("  12 - Rating evolution over time")
print("  13 - Director loyalty chart")
print("  14 - Director network")
print()