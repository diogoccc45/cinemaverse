"""
SCRIPT 08b - Personalised Film Recommendations
===============================================
Uses cosine similarity between your taste profile
and candidate films to recommend what to watch next.

Two recommendation pools:
  1. Your watchlist (248 films you already saved)
  2. Discovery candidates (fetched in 08a)
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, '..')
DATA_DIR   = os.path.join(ROOT_DIR, 'data', 'processed')
RAW_DIR    = os.path.join(ROOT_DIR, 'data', 'raw')
OUT_DIR    = os.path.join(ROOT_DIR, 'output')
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
    'grid.linewidth':    0.6,
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
BLUE   = '#4488ff'

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
print("Loading data...")

enriched   = pd.read_csv(os.path.join(DATA_DIR, 'movies_enriched.csv'))
candidates = pd.read_csv(os.path.join(DATA_DIR, 'candidates.csv'))
watchlist  = pd.read_csv(os.path.join(RAW_DIR, 'watchlist.csv'))

# Enrich watchlist with TMDb data from candidates where possible
watchlist_enriched = watchlist.merge(
    candidates[['Name', 'Year', 'genres', 'directors', 'cast',
                'countries', 'runtime', 'tmdb_rating', 'overview']],
    on=['Name', 'Year'], how='left'
)
# Fill any watchlist films not in candidates with empty strings
for col in ['genres', 'directors', 'cast', 'countries']:
    watchlist_enriched[col] = watchlist_enriched[col].fillna('')

print(f"  {len(enriched)} watched films")
print(f"  {len(watchlist_enriched)} watchlist films")
print(f"  {len(candidates)} discovery candidates")

# ─────────────────────────────────────────────────────────────
# BUILD FEATURE VECTORS
# Same feature space as script 06 so everything is comparable
# ─────────────────────────────────────────────────────────────
print("\nBuilding feature vectors...")

# Get top genres and directors from watched films
top_genres   = enriched['genres'].dropna().str.split('|').explode().value_counts().head(20).index.tolist()
top_dirs     = enriched['directors'].dropna().str.split('|').explode().value_counts().head(30).index.tolist()
top_countries = enriched['countries'].dropna().str.split('|').explode().value_counts().head(10).index.tolist()

year_min = enriched['Year'].min()
year_max = enriched['Year'].max()

def build_vector(row, rating=None):
    """
    Converts a film's metadata into a numeric vector.
    Same logic as script 06 for consistency.
    """
    vec = []

    # Numeric features
    vec.append((rating / 5.0) if rating else 0.5)
    year = row.get('Year', 2000) if pd.notna(row.get('Year', None)) else 2000
    vec.append((year - year_min) / max(year_max - year_min, 1))
    vec.append(((year // 10 * 10) - 1920) / 100)
    runtime = row.get('runtime', 100) or 100
    vec.append(float(runtime) / 240)
    tmdb = row.get('tmdb_rating', 0) or 0
    vec.append(float(tmdb) / 10.0)

    # Genre flags (weighted 1.5x)
    genres = str(row.get('genres', ''))
    for g in top_genres:
        vec.append(1.5 if g in genres else 0.0)

    # Country flags
    countries = str(row.get('countries', ''))
    for c in top_countries:
        vec.append(1.0 if c in countries else 0.0)

    # Director flags (weighted 2x)
    directors = str(row.get('directors', ''))
    for d in top_dirs:
        vec.append(2.0 if d in directors else 0.0)

    return np.array(vec, dtype=float)

# Build taste profile from films rated 4.5 or 5 stars
loved = enriched[enriched['Rating'] >= 4.5]
print(f"  Building taste profile from {len(loved)} loved films (4.5+ stars)...")

loved_vectors = np.array([
    build_vector(row, rating=row['Rating'])
    for _, row in loved.iterrows()
])

# Taste profile = weighted average of loved film vectors
# Weight by rating so 5-star films count more than 4.5-star
weights = loved['Rating'].values
taste_profile = np.average(loved_vectors, axis=0, weights=weights)
taste_profile = np.nan_to_num(taste_profile, nan=0.0)

# ─────────────────────────────────────────────────────────────
# COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────
from sklearn.metrics.pairwise import cosine_similarity

def recommend(pool_df, pool_name, top_n=20):
    """
    Scores each film in the pool by cosine similarity
    to the taste profile, returns top N recommendations.
    """
    vectors = np.array([
        build_vector(row) for _, row in pool_df.iterrows()
    ])
    # Replace any NaN with 0 — some watchlist films have missing metadata
    vectors = np.nan_to_num(vectors, nan=0.0)

    # cosine_similarity expects 2D arrays
    profile_2d = taste_profile.reshape(1, -1)
    sims = cosine_similarity(profile_2d, vectors)[0]

    pool_df = pool_df.copy()
    pool_df['similarity'] = sims
    pool_df['similarity_pct'] = (sims * 100).round(1)

    top = pool_df.nlargest(top_n, 'similarity').reset_index(drop=True)
    print(f"\n  Top {top_n} from {pool_name}:")
    for i, row in top.head(10).iterrows():
        name = row['Name'][:40]
        genres = str(row.get('genres', '')).replace('|', ', ')[:35]
        print(f"    {i+1:2d}. {name:<42} {row['similarity_pct']:5.1f}%  {genres}")

    return top

print("\nComputing recommendations...")
recs_watchlist   = recommend(watchlist_enriched, "Watchlist", top_n=30)
recs_discoveries = recommend(candidates, "Discoveries", top_n=30)

# ─────────────────────────────────────────────────────────────
# CHART 1 - TOP WATCHLIST RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
print("\nGenerating charts...")

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('My Watchlist — Ranked by Match to My Taste', color=GOLD, fontsize=14)

top_wl = recs_watchlist.head(20)
colors = [GREEN if s >= 90 else GOLD if s >= 80 else TEAL
          for s in top_wl['similarity_pct']]

bars = ax.barh(
    [f"{row['Name'][:38]}  ({int(row['Year'])})" for _, row in top_wl.iterrows()],
    top_wl['similarity_pct'],
    color=colors[::-1], zorder=3
)
ax.set_xlabel('Match to my taste (%)')
ax.set_title('Higher = more aligned with films I rated 4.5-5 stars')
ax.set_facecolor('#111118')
ax.set_xlim(0, 105)

for bar, (_, row) in zip(bars, top_wl[::-1].iterrows()):
    genres = str(row.get('genres', '')).replace('|', ' · ')[:30]
    ax.text(bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{row['similarity_pct']}%  {genres}",
            va='center', fontsize=7.5, color='#9999b8')

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=GREEN, label='>= 90% match'),
    Patch(color=GOLD,  label='>= 80% match'),
    Patch(color=TEAL,  label='< 80% match'),
], facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '27_watchlist_recommendations.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/27_watchlist_recommendations.png")

# ─────────────────────────────────────────────────────────────
# CHART 2 - TOP DISCOVERY RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('Discovery Recommendations — Films You Have Not Seen or Saved', color=GOLD, fontsize=14)

top_disc = recs_discoveries.head(20)
colors_d = [GREEN if s >= 90 else GOLD if s >= 80 else TEAL
            for s in top_disc['similarity_pct']]

bars = ax.barh(
    [f"{row['Name'][:38]}  ({int(row['Year'])})" for _, row in top_disc.iterrows()],
    top_disc['similarity_pct'],
    color=colors_d[::-1], zorder=3
)
ax.set_xlabel('Match to my taste (%)')
ax.set_title('Films outside your watchlist that match your taste profile')
ax.set_facecolor('#111118')
ax.set_xlim(0, 105)

for bar, (_, row) in zip(bars, top_disc[::-1].iterrows()):
    genres = str(row.get('genres', '')).replace('|', ' · ')[:30]
    tmdb   = row.get('tmdb_rating', 0)
    ax.text(bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{row['similarity_pct']}%  TMDb {tmdb:.1f}  {genres}",
            va='center', fontsize=7.5, color='#9999b8')

ax.legend(handles=[
    Patch(color=GREEN, label='>= 90% match'),
    Patch(color=GOLD,  label='>= 80% match'),
    Patch(color=TEAL,  label='< 80% match'),
], facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '28_discovery_recommendations.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/28_discovery_recommendations.png")

# ─────────────────────────────────────────────────────────────
# CHART 3 - TASTE PROFILE BREAKDOWN
# What does your taste profile actually look like?
# ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('My Taste Profile — What the Recommendation Model Learned', color=GOLD, fontsize=13)

# Genre weights in the taste profile
n_numeric   = 5
n_genres    = len(top_genres)
n_countries = len(top_countries)

genre_weights = taste_profile[n_numeric:n_numeric + n_genres]
genre_series  = pd.Series(genre_weights, index=top_genres).sort_values(ascending=True)

colors_g = [GREEN if v >= genre_series.quantile(0.75) else
            GOLD   if v >= genre_series.median() else TEAL
            for v in genre_series]
ax1.barh(genre_series.index, genre_series.values, color=colors_g, zorder=3)
ax1.set_title('Genre weights in my taste profile')
ax1.set_xlabel('Weight (higher = stronger preference)')
ax1.set_facecolor('#111118')

# Director weights
dir_weights = taste_profile[n_numeric + n_genres + n_countries:]
dir_series  = pd.Series(dir_weights, index=top_dirs).sort_values(ascending=False).head(15).sort_values()

colors_d2 = [GREEN if v >= dir_series.quantile(0.75) else
             GOLD   if v >= dir_series.median() else TEAL
             for v in dir_series]
ax2.barh(dir_series.index, dir_series.values, color=colors_d2, zorder=3)
ax2.set_title('Director weights in my taste profile\n(top 15)')
ax2.set_xlabel('Weight (higher = stronger preference)')
ax2.set_facecolor('#111118')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '29_taste_profile.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/29_taste_profile.png")

# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────
recs_watchlist.to_csv(os.path.join(DATA_DIR, 'recs_watchlist.csv'),
                      index=False, encoding='utf-8')
recs_discoveries.to_csv(os.path.join(DATA_DIR, 'recs_discoveries.csv'),
                        index=False, encoding='utf-8')

print()
print("=" * 55)
print("  DONE!")
print("=" * 55)
print(f"  27 - Watchlist ranked by match")
print(f"  28 - Discovery recommendations")
print(f"  29 - Taste profile breakdown")
print()
print(f"  Top 5 watchlist picks:")
for i, row in recs_watchlist.head(5).iterrows():
    print(f"    {i+1}. {row['Name']} ({int(row['Year'])}) — {row['similarity_pct']}% match")
print()
print(f"  Top 5 discoveries:")
for i, row in recs_discoveries.head(5).iterrows():
    print(f"    {i+1}. {row['Name']} ({int(row['Year'])}) — {row['similarity_pct']}% match")
print()
print("  Results saved to data/processed/")
