"""
SCRIPT 09b - Semantic Recommendations via Synopsis Embeddings
=============================================================
Uses sentence-transformers to embed film synopses into 384-dim vectors.
Finds watchlist films semantically similar to films you loved (4.5+ stars).

This solves the cold start problem that affects the Random Forest:
the model doesn't need your rating history for a film — just its synopsis.

How to run:
    pip install sentence-transformers
    python scripts/09b_semantic_recommendations.py
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
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
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

# ─────────────────────────────────────────────────────────────
# STEP 1 - LOAD DATA
# ─────────────────────────────────────────────────────────────

enriched   = pd.read_csv(os.path.join(DATA_DIR, 'movies_enriched.csv'))
watchlist  = pd.read_csv(os.path.join(DATA_DIR, 'watchlist_enriched.csv'))
candidates = pd.read_csv(os.path.join(DATA_DIR, 'candidates.csv'))

# Films with synopsis
enriched_with_overview = enriched[
    enriched['overview'].notna() & (enriched['overview'].str.len() > 20)
].copy()

watchlist_with_overview = watchlist[
    watchlist['overview'].notna() & (watchlist['overview'].str.len() > 20)
].copy()

candidates_with_overview = candidates[
    candidates['overview'].notna() & (candidates['overview'].str.len() > 20)
].copy()

print(f"  Watched films with synopsis:    {len(enriched_with_overview)}")
print(f"  Watchlist films with synopsis:  {len(watchlist_with_overview)}")
print(f"  Discovery films with synopsis:  {len(candidates_with_overview)}")

# Films you loved (4.5+ stars) — the "taste anchors"
loved = enriched_with_overview[enriched_with_overview['Rating'] >= 4.5].copy()
print(f"  Films you loved (4.5+):         {len(loved)}")

# ─────────────────────────────────────────────────────────────
# STEP 2 - BUILD EMBEDDINGS
# sentence-transformers encodes each synopsis into a 384-dim vector
# Films with similar meaning end up close in this vector space
# ─────────────────────────────────────────────────────────────

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# all-MiniLM-L6-v2: small, fast, high quality — perfect for CPU
model = SentenceTransformer('all-MiniLM-L6-v2')
print("  Model loaded!")

# Check if embeddings are already cached
EMBED_CACHE = os.path.join(DATA_DIR, 'synopsis_embeddings.npz')

if os.path.exists(EMBED_CACHE):
    print("\nStep 3/5 - Loading cached embeddings...")
    cached = np.load(EMBED_CACHE, allow_pickle=True)
    loved_embeddings      = cached['loved']
    watchlist_embeddings  = cached['watchlist']
    candidate_embeddings  = cached['candidates']
    print(f"  Loaded from cache!")
else:
    print("  Encoding loved films...")
    loved_embeddings = model.encode(
        loved['overview'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )

    print("  Encoding watchlist films...")
    watchlist_embeddings = model.encode(
        watchlist_with_overview['overview'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )

    print("  Encoding discovery candidates...")
    candidate_embeddings = model.encode(
        candidates_with_overview['overview'].tolist(),
        show_progress_bar=True,
        batch_size=32
    )

    # Cache for future runs
    np.savez(EMBED_CACHE,
             loved=loved_embeddings,
             watchlist=watchlist_embeddings,
             candidates=candidate_embeddings)
    print("  Embeddings cached for future runs!")

# ─────────────────────────────────────────────────────────────
# STEP 4 - COMPUTE SEMANTIC SIMILARITY
# For each candidate film, compute its avg cosine similarity
# to all films you loved — weighted by your rating
# ─────────────────────────────────────────────────────────────

# Weights: 5-star films count more than 4.5-star
rating_weights = loved['Rating'].values
rating_weights = rating_weights / rating_weights.sum()

def semantic_score(candidate_emb, loved_embs, weights):
    """
    Computes weighted average cosine similarity between
    a candidate film and all films you loved.
    Higher = more semantically similar to films you adore.
    """
    sims = cosine_similarity(
        candidate_emb.reshape(1, -1),
        loved_embs
    )[0]
    return float(np.average(sims, weights=weights))

# Score watchlist films
print("  Scoring watchlist...")
watchlist_scores = []
for i, emb in enumerate(watchlist_embeddings):
    score = semantic_score(emb, loved_embeddings, rating_weights)
    watchlist_scores.append(score)

watchlist_with_overview = watchlist_with_overview.copy()
watchlist_with_overview['semantic_score'] = watchlist_scores
watchlist_with_overview['semantic_pct']   = (
    pd.Series(watchlist_scores).rank(pct=True) * 100
).round(1).values

# Score discovery candidates
candidate_scores = []
for i, emb in enumerate(candidate_embeddings):
    score = semantic_score(emb, loved_embeddings, rating_weights)
    candidate_scores.append(score)

candidates_with_overview = candidates_with_overview.copy()
candidates_with_overview['semantic_score'] = candidate_scores
candidates_with_overview['semantic_pct']   = (
    pd.Series(candidate_scores).rank(pct=True) * 100
).round(1).values

# Save results
watchlist_with_overview.to_csv(
    os.path.join(DATA_DIR, 'watchlist_semantic.csv'), index=False)
candidates_with_overview.to_csv(
    os.path.join(DATA_DIR, 'candidates_semantic.csv'), index=False)

# ─────────────────────────────────────────────────────────────
# STEP 5 - VISUALISE
# ─────────────────────────────────────────────────────────────

# ── CHART 1: Top semantic watchlist recommendations ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle(
    'Semantic Recommendations — based on synopsis similarity to films I loved',
    color=GOLD, fontsize=13
)

for ax, pool_df, title in [
    (ax1, watchlist_with_overview,  'Watchlist — semantic match'),
    (ax2, candidates_with_overview, 'Discoveries — semantic match'),
]:
    top = pool_df.nlargest(15, 'semantic_score')
    scores = top['semantic_score'].values
    # Normalise to 0-100 for display
    score_pct = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9) * 100

    colors = [GREEN if s >= 75 else GOLD if s >= 50 else TEAL for s in score_pct]

    bars = ax.barh(
        [f"{row['Name'][:38]}  ({int(row['Year'])})" for _, row in top.iterrows()],
        score_pct[::-1],
        color=colors[::-1], zorder=3
    )

    for bar, (_, row) in zip(bars, top[::-1].iterrows()):
        genres = str(row.get('genres', '')).replace('|', ' · ')[:28]
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{genres}",
                va='center', fontsize=7, color='#9999b8')

    ax.set_xlabel('Semantic similarity score (relative)')
    ax.set_title(title)
    ax.set_facecolor('#111118')
    ax.set_xlim(0, 130)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '35_semantic_recommendations.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/35_semantic_recommendations.png")

# ── CHART 2: Most similar films to your top 5-star films ──

top5 = enriched_with_overview[enriched_with_overview['Rating'] == 5.0].head(8)
top5_embs = model.encode(top5['overview'].tolist(), show_progress_bar=False)

fig, axes = plt.subplots(2, 4, figsize=(20, 11))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle(
    'If you loved these... you should watch these (semantic similarity)',
    color=GOLD, fontsize=14, y=0.98
)

for ax, (_, film_row), film_emb in zip(axes.flat, top5.iterrows(), top5_embs):
    sims    = cosine_similarity(film_emb.reshape(1, -1), watchlist_embeddings)[0]
    top3_idx = sims.argsort()[::-1][:3]
    top3     = watchlist_with_overview.iloc[top3_idx]

    ax.set_facecolor('#0d0d18')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a4a')
        spine.set_linewidth(1.2)

    # Header — loved film
    film_name = film_row['Name']

    ax.text(0.5, 0.91, film_name,
            transform=ax.transAxes, ha='center', va='center',
            fontsize=10, color=GOLD, fontweight='bold')
    ax.text(0.5, 0.81, f"5 stars   {int(film_row['Year'])}",
            transform=ax.transAxes, ha='center', va='center',
            fontsize=8, color='#888899')

    # Divider
    ax.axhline(y=0.76, color='#2a2a4a', linewidth=1.0)

    # Recommendations
    positions = [0.60, 0.38, 0.16]
    for i, ((_, rec), pos) in enumerate(zip(top3.iterrows(), positions)):
        sim_score = sims[top3_idx[i]]
        rec_name  = rec['Name']
        genres    = str(rec.get('genres', '')).split('|')[0]

        # Rank badge colour
        badge_color = [GREEN, GOLD, TEAL][i]

        ax.text(0.08, pos + 0.06, f"{i+1}",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color=badge_color, fontweight='bold')
        ax.text(0.20, pos + 0.06, rec_name,
                transform=ax.transAxes, ha='left', va='center',
                fontsize=8, color='#e8e8f0', fontweight='bold')
        ax.text(0.20, pos - 0.02, f"{genres}  |  {int(rec['Year'])}  |  sim {sim_score:.3f}",
                transform=ax.transAxes, ha='left', va='center',
                fontsize=7, color='#666688')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUT_DIR, '36_semantic_similar_pairs.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/36_semantic_similar_pairs.png")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print(f"  35 - Semantic recommendations (watchlist + discoveries)")
print(f"  36 - Film-by-film similarity pairs")
print()
print("  Top 10 watchlist films by semantic similarity:")
top10_wl = watchlist_with_overview.nlargest(10, 'semantic_score')
for i, (_, row) in enumerate(top10_wl.iterrows()):
    print(f"    {i+1:2d}. {row['Name'][:45]:<47} ({int(row['Year'])})")
print()
print("  Top 10 discoveries by semantic similarity:")
top10_disc = candidates_with_overview.nlargest(10, 'semantic_score')
for i, (_, row) in enumerate(top10_disc.iterrows()):
    print(f"    {i+1:2d}. {row['Name'][:45]:<47} ({int(row['Year'])})")
print()
