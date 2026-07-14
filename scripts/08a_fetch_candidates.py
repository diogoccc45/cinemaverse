"""
SCRIPT 08a - Fetch candidate films from TMDb
=============================================
Fetches ~500 popular films from your favourite genres
that you haven't watched or added to your watchlist yet.
These become the "discovery" pool for recommendations.
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import requests
import json
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, '..')
DATA_DIR   = os.path.join(ROOT_DIR, 'data', 'processed')
RAW_DIR    = os.path.join(ROOT_DIR, 'data', 'raw')

# Load API key from .env
env_path = os.path.join(ROOT_DIR, '.env')
API_KEY  = None
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('TMDB_KEY='):
                API_KEY = line.split('=', 1)[1].strip()

if not API_KEY:
    API_KEY = input("Paste your TMDb API key: ").strip()

BASE_URL = 'https://api.themoviedb.org/3'

def api_get(endpoint, params={}):
    params = dict(params)
    params['api_key'] = API_KEY
    try:
        r = requests.get(f'{BASE_URL}{endpoint}', params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  API error: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# LOAD WHAT WE ALREADY HAVE
# ─────────────────────────────────────────────────────────────

watched = pd.read_csv(os.path.join(RAW_DIR, 'watched.csv'))
watchlist = pd.read_csv(os.path.join(RAW_DIR, 'watchlist.csv'))
enriched = pd.read_csv(os.path.join(DATA_DIR, 'movies_enriched.csv'))

already_known = set(watched['Name'].str.lower()) | set(watchlist['Name'].str.lower())
print(f"  Already know {len(already_known)} films (watched + watchlist)")

# Top genres by avg rating — fetch films from these first
genres_exp = enriched[enriched['genres'].notna()].assign(
    genre=enriched['genres'].str.split('|')
).explode('genre')
top_genres_rated = genres_exp.groupby('genre')['Rating'].mean().nlargest(15)
print(f"  Top genres to fetch: {list(top_genres_rated.index)}")

# TMDb genre IDs
GENRE_IDS = {
    'Drama': 18, 'Comedy': 35, 'Thriller': 53, 'Crime': 80,
    'Science Fiction': 878, 'Horror': 27, 'Animation': 16,
    'Documentary': 99, 'Mystery': 9648, 'Action': 28,
    'Adventure': 12, 'Romance': 10749, 'Fantasy': 14,
    'History': 36, 'War': 10752, 'Music': 10402,
}

# ─────────────────────────────────────────────────────────────
# FETCH POPULAR FILMS BY GENRE
# ─────────────────────────────────────────────────────────────

candidates = []
seen_ids = set()

for genre in top_genres_rated.index:
    if genre not in GENRE_IDS:
        continue

    genre_id = GENRE_IDS[genre]
    print(f"  Fetching {genre}...")

    # Fetch 6 pages per genre = ~120 films, max 60 kept per genre
    genre_count = 0
    for page in range(1, 7):
        result = api_get('/discover/movie', {
            'with_genres':          genre_id,
            'sort_by':              'vote_average.desc',
            'vote_count.gte':       200,   # reasonably well-known films
            'language':             'en-US',
            'page':                 page,
        })
        time.sleep(0.15)

        if not result or not result.get('results'):
            break

        for film in result['results']:
            if film['id'] in seen_ids:
                continue
            if film['title'].lower() in already_known:
                continue
            if film.get('vote_average', 0) < 5.5:
                continue

            if genre_count >= 60:
                break
            seen_ids.add(film['id'])
            genre_count += 1
            candidates.append({
                'tmdb_id':    film['id'],
                'Name':       film['title'],
                'Year':       int(film.get('release_date', '2000')[:4]) if film.get('release_date') else 2000,
                'overview':   film.get('overview', ''),
                'tmdb_rating': film.get('vote_average', 0),
                'tmdb_votes':  film.get('vote_count', 0),
                'genre_ids':  str(film.get('genre_ids', [])),
            })

print(f"\n  {len(candidates)} candidate films found")

# ─────────────────────────────────────────────────────────────
# FETCH FULL DETAILS FOR EACH CANDIDATE
# ─────────────────────────────────────────────────────────────

enriched_candidates = []

for i, film in enumerate(candidates):
    print(f"  [{i+1}/{len(candidates)}] {film['Name']}", end='', flush=True)

    result = api_get(f"/movie/{film['tmdb_id']}", {
        'append_to_response': 'credits',
        'language': 'en-US'
    })
    time.sleep(0.15)

    if not result:
        print(" - skipped")
        continue

    genres    = '|'.join(g['name'] for g in result.get('genres', []))
    directors = '|'.join(p['name'] for p in result.get('credits', {}).get('crew', [])
                         if p['job'] == 'Director')
    cast      = '|'.join(p['name'] for p in result.get('credits', {}).get('cast', [])[:5])
    countries = '|'.join(c['iso_3166_1'] for c in result.get('production_countries', []))

    print(f" - {directors.split('|')[0] if directors else '?'} | {genres[:30]}")

    enriched_candidates.append({
        **film,
        'genres':    genres,
        'directors': directors,
        'cast':      cast,
        'countries': countries,
        'runtime':   result.get('runtime', 0),
        'tagline':   result.get('tagline', ''),
        'Rating':    None,  # not watched yet
    })

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
df_candidates = pd.DataFrame(enriched_candidates)
out_path = os.path.join(DATA_DIR, 'candidates.csv')
df_candidates.to_csv(out_path, index=False, encoding='utf-8')

print()
print("=" * 50)
print(f"  {len(df_candidates)} candidates saved.")
print(f"  Saved to data/processed/candidates.csv")