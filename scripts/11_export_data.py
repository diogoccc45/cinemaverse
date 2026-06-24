"""
SCRIPT 11 - Export all processed data to data.json
====================================================
Consolidates all processed data into a single JSON file
for use in the Streamlit dashboard.

How to run:
    python scripts/11_export_data.py
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, '..')
DATA_DIR   = os.path.join(ROOT_DIR, 'data', 'processed')
RAW_DIR    = os.path.join(ROOT_DIR, 'data', 'raw')
OUT_FILE   = os.path.join(ROOT_DIR, 'data.json')

print("Loading data...")

# ─────────────────────────────────────────────────────────────
# LOAD ALL PROCESSED DATA
# ─────────────────────────────────────────────────────────────
enriched   = pd.read_csv(os.path.join(DATA_DIR, 'movies_enriched.csv'))
watchlist  = pd.read_csv(os.path.join(DATA_DIR, 'watchlist_enriched.csv'))
diary      = pd.read_csv(os.path.join(RAW_DIR,  'diary.csv'))
reviews    = pd.read_csv(os.path.join(RAW_DIR,  'reviews.csv'))
diary['Date'] = pd.to_datetime(diary['Date'])

recs_wl   = pd.read_csv(os.path.join(DATA_DIR, 'recs_watchlist.csv'))
recs_disc = pd.read_csv(os.path.join(DATA_DIR, 'recs_discoveries.csv'))

try:
    sem_wl   = pd.read_csv(os.path.join(DATA_DIR, 'watchlist_semantic.csv'))
    sem_disc = pd.read_csv(os.path.join(DATA_DIR, 'candidates_semantic.csv'))
    has_semantic = True
except FileNotFoundError:
    has_semantic = False
    print("  No semantic recommendations found — skipping")

try:
    with open(os.path.join(DATA_DIR, 'network_comparison.json'), 'r') as f:
        network_metrics = json.load(f)
except FileNotFoundError:
    network_metrics = {}

try:
    embeddings = pd.read_csv(os.path.join(DATA_DIR, 'embeddings.csv'))
    has_embeddings = True
except FileNotFoundError:
    has_embeddings = False

print(f"  {len(enriched)} watched films")
print(f"  {len(watchlist)} watchlist films")

# ─────────────────────────────────────────────────────────────
# BUILD DATA DICT
# ─────────────────────────────────────────────────────────────
print("Building data.json...")

def clean_df(df, cols=None):
    if cols:
        df = df[cols]
    df = df.copy()
    df = df.where(pd.notna(df), None)
    return df.to_dict('records')

data = {}

# Overview stats
data['stats'] = {
    'total_watched':   int(len(enriched)),
    'total_rated':     int(enriched['Rating'].notna().sum()),
    'avg_rating':      round(float(enriched['Rating'].mean()), 2),
    'total_reviews':   int(reviews['Review'].notna().sum()),
    'five_star_count': int((enriched['Rating'] == 5.0).sum()),
    'half_star_count': int((enriched['Rating'] == 0.5).sum()),
    'watchlist_count': int(len(watchlist)),
    'oldest_film':     int(enriched['Year'].min()),
    'newest_film':     int(enriched['Year'].max()),
    'first_log':       str(diary['Date'].min().date()),
    'last_log':        str(diary['Date'].max().date()),
}

# Rating distribution
dist = enriched['Rating'].value_counts().sort_index()
data['rating_distribution'] = {str(k): int(v) for k, v in dist.items()}

# Monthly activity
diary['yearmonth'] = diary['Date'].dt.to_period('M').astype(str)
monthly = diary.groupby('yearmonth').size().reset_index(name='count')
data['monthly_activity'] = clean_df(monthly)

# Decades
enriched['decade'] = (enriched['Year'] // 10 * 10).astype(int)
decade_data = enriched.groupby('decade').agg(
    count=('Name', 'count'),
    avg_rating=('Rating', 'mean')
).reset_index()
decade_data['avg_rating'] = decade_data['avg_rating'].round(2)
data['decades'] = clean_df(decade_data)

# Top genres
genres_exp = enriched[enriched['genres'].notna()].assign(
    genre=enriched['genres'].str.split('|')
).explode('genre')
genre_stats = genres_exp.groupby('genre').agg(
    count=('Name', 'count'),
    avg_rating=('Rating', 'mean')
).query('count >= 5').sort_values('count', ascending=False).reset_index()
genre_stats['avg_rating'] = genre_stats['avg_rating'].round(2)
data['genres'] = clean_df(genre_stats)

# Top directors
dirs_exp = enriched[enriched['directors'].notna()].assign(
    director=enriched['directors'].str.split('|')
).explode('director')
dir_stats = dirs_exp.groupby('director').agg(
    count=('Name', 'count'),
    avg_rating=('Rating', 'mean')
).query('count >= 3').sort_values('avg_rating', ascending=False).reset_index()
dir_stats['avg_rating'] = dir_stats['avg_rating'].round(2)
data['directors'] = clean_df(dir_stats.head(50))

# Top actors
cast_exp = enriched[enriched['cast'].notna()].assign(
    actor=enriched['cast'].str.split('|')
).explode('actor')
actor_stats = cast_exp.groupby('actor').agg(
    count=('Name', 'count'),
    avg_rating=('Rating', 'mean')
).query('count >= 5').sort_values('count', ascending=False).reset_index()
actor_stats['avg_rating'] = actor_stats['avg_rating'].round(2)
data['actors'] = clean_df(actor_stats.head(50))

# Top production companies
if 'production_companies' in enriched.columns:
    comp_exp = enriched[enriched['production_companies'].notna()].assign(
        company=enriched['production_companies'].str.split('|')
    ).explode('company')
    comp_exp = comp_exp[comp_exp['company'].str.strip() != '']
    comp_stats = comp_exp.groupby('company').agg(
        count=('Name', 'count'),
        avg_rating=('Rating', 'mean')
    ).query('count >= 3').sort_values('avg_rating', ascending=False).reset_index()
    comp_stats['avg_rating'] = comp_stats['avg_rating'].round(2)
    data['companies'] = clean_df(comp_stats.head(30))

# Films data 
film_cols = ['Name', 'Year', 'Rating', 'genres', 'directors',
             'countries', 'runtime', 'tmdb_rating', 'overview']
films_clean = enriched[[c for c in film_cols if c in enriched.columns]].copy()
films_clean['decade'] = (films_clean['Year'] // 10 * 10).astype(int)
data['films'] = clean_df(films_clean)

# Galaxy embeddings
if has_embeddings:
    emb_cols = ['Name', 'Year', 'Rating', 'genres', 'cluster', 'umap_x', 'umap_y']
    emb_clean = embeddings[[c for c in emb_cols if c in embeddings.columns]].copy()
    emb_clean['umap_x'] = emb_clean['umap_x'].round(4)
    emb_clean['umap_y'] = emb_clean['umap_y'].round(4)
    data['galaxy'] = clean_df(emb_clean)

# Reviews
reviews_clean = reviews[reviews['Review'].notna()].copy()
reviews_clean['review_length'] = reviews_clean['Review'].str.len()
data['reviews'] = clean_df(
    reviews_clean[['Name', 'Year', 'Rating', 'Review', 'review_length']].head(200)
)

# Me vs crowd 
compare = enriched[(enriched['Rating'].notna()) & (enriched['tmdb_rating'] > 0)].copy()
compare['tmdb_5'] = (compare['tmdb_rating'] / 2).round(2)
compare['diff']   = (compare['Rating'] - compare['tmdb_5']).round(2)
data['me_vs_crowd'] = {
    'avg_bias':     round(float(compare['diff'].mean()), 3),
    'higher_count': int((compare['diff'] > 0).sum()),
    'lower_count':  int((compare['diff'] < 0).sum()),
    'equal_count':  int((compare['diff'] == 0).sum()),
    'loved_more': clean_df(
        compare[compare['diff'] > 1.5].nlargest(10, 'diff')
        [['Name', 'Year', 'Rating', 'tmdb_5', 'diff']]
    ),
    'hated_more': clean_df(
        compare[compare['diff'] < -1.5].nsmallest(10, 'diff')
        [['Name', 'Year', 'Rating', 'tmdb_5', 'diff']]
    ),
}

# Recommendations
data['recommendations'] = {
    'watchlist_cosine': clean_df(
        recs_wl[['Name', 'Year', 'similarity_pct', 'genres', 'directors']].head(30)
    ),
    'discoveries_cosine': clean_df(
        recs_disc[['Name', 'Year', 'similarity_pct', 'genres', 'directors']].head(30)
    ),
}
if has_semantic:
    data['recommendations']['watchlist_semantic'] = clean_df(
        sem_wl.nlargest(30, 'semantic_score')[['Name', 'Year', 'semantic_score', 'genres', 'directors']]
    )
    data['recommendations']['discoveries_semantic'] = clean_df(
        sem_disc.nlargest(30, 'semantic_score')[['Name', 'Year', 'semantic_score', 'genres', 'directors']]
    )

# Network metrics 
data['network'] = network_metrics

# Network graph — director network for visualisation
print("Building director network graph...")
try:
    from itertools import combinations
    import networkx as nx

    # Build actor -> directors mapping
    film_data = enriched[enriched['directors'].notna() & enriched['cast'].notna()].copy()
    film_data['dir_list']  = film_data['directors'].str.split('|')
    film_data['cast_list'] = film_data['cast'].str.split('|')

    # Directors with >= 3 films
    top_dirs_set = set(
        dirs_exp.groupby('director').size()[
            dirs_exp.groupby('director').size() >= 3
        ].index
    )

    # Build graph
    G = nx.Graph()
    actor_to_dirs = {}
    for _, row in film_data.iterrows():
        for actor in row['cast_list']:
            for director in row['dir_list']:
                actor_to_dirs.setdefault(actor, set()).add(director)

    for actor, dirs in actor_to_dirs.items():
        dirs = [d for d in dirs if d in top_dirs_set]
        if len(dirs) >= 2:
            for d1, d2 in combinations(sorted(dirs), 2):
                if G.has_edge(d1, d2):
                    G[d1][d2]['weight'] += 1
                else:
                    G.add_edge(d1, d2, weight=1)

    # Keep only edges with weight >= 2
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 2])
    G.remove_nodes_from(list(nx.isolates(G)))

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # Node attributes
    dir_avg  = dirs_exp.groupby('director')['Rating'].mean()
    dir_cnt  = dirs_exp.groupby('director').size()

    nodes = []
    for node in G.nodes():
        x, y = pos[node]
        nodes.append({
            'id':         node,
            'x':          round(float(x), 4),
            'y':          round(float(y), 4),
            'avg_rating': round(float(dir_avg.get(node, 3.0)), 2),
            'films':      int(dir_cnt.get(node, 1)),
        })

    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({
            'source': u,
            'target': v,
            'weight': int(d['weight']),
            'x0': round(float(pos[u][0]), 4),
            'y0': round(float(pos[u][1]), 4),
            'x1': round(float(pos[v][0]), 4),
            'y1': round(float(pos[v][1]), 4),
        })

    data['network_graph'] = {'nodes': nodes, 'edges': edges}
    print(f"  Network graph: {len(nodes)} nodes, {len(edges)} edges")

except Exception as e:
    print(f"  Network graph skipped: {e}")
    data['network_graph'] = {'nodes': [], 'edges': []}

# 5-star and 0.5-star films
data['five_star'] = clean_df(
    enriched[enriched['Rating'] == 5.0][['Name', 'Year', 'genres', 'directors']].sort_values('Year')
)
data['half_star'] = clean_df(
    enriched[enriched['Rating'] == 0.5][['Name', 'Year', 'genres', 'directors']].sort_values('Year')
)

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, separators=(',', ':'))

size_mb = os.path.getsize(OUT_FILE) / 1024 / 1024
print()
print("=" * 50)
print(f"  Saved: data.json ({size_mb:.1f} MB)")
print()
print("  Contents:")
for key, val in data.items():
    if isinstance(val, list):
        print(f"    {key}: {len(val)} records")
    elif isinstance(val, dict):
        print(f"    {key}: {list(val.keys())[:5]}")