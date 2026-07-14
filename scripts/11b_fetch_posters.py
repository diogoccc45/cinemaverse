"""
SCRIPT 11b - Fetch poster paths from TMDb
==========================================
Goes through movies_enriched.csv and fetches the best-rated
poster for each film from TMDb.

How to run:
    python scripts/11b_fetch_posters.py
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
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')

CACHE_FILE = os.path.join(DATA_DIR, 'posters_cache.json')
OUTPUT_FILE = os.path.join(DATA_DIR, 'movies_enriched.csv')

# Load API key
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
    except:
        return None

# Load cache
cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    print(f"Cache loaded: {len(cache)} posters")

# Load enriched data
df = pd.read_csv(OUTPUT_FILE)
print(f"Films: {len(df)}")

poster_paths = []

for i, row in df.iterrows():
    tmdb_id = row.get('tmdb_id')
    if not tmdb_id or str(tmdb_id) == 'nan':
        poster_paths.append(None)
        continue

    key = str(int(float(tmdb_id)))

    if key in cache:
        poster_paths.append(cache[key])
        continue

    print(f"[{i+1}/{len(df)}] {row['Name']}", end='', flush=True)

    # Get all posters and pick the best rated one
    result = api_get(f'/movie/{key}/images', {'include_image_language': 'en,null'})
    time.sleep(0.15)

    if result and result.get('posters'):
        # Sort by vote average descending
        posters = sorted(result['posters'], key=lambda x: x.get('vote_average', 0), reverse=True)
        best = posters[0].get('file_path')
        print(f" -> {best}")
    else:
        # Fallback to default poster
        mov = api_get(f'/movie/{key}')
        time.sleep(0.15)
        best = mov.get('poster_path') if mov else None
        print(f" -> fallback {best}")

    cache[key] = best
    poster_paths.append(best)

    if (i + 1) % 50 == 0:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
        print(f"\n  -- Cache saved ({i+1} done) --\n")

# Save cache
with open(CACHE_FILE, 'w', encoding='utf-8') as f:
    json.dump(cache, f)

# Add to dataframe
df['poster_path'] = poster_paths
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

has_poster = sum(1 for p in poster_paths if p)
print()
print("=" * 50)
print(f"  {has_poster}/{len(df)} films have posters")
print(f"  Saved to movies_enriched.csv")
print("=" * 50)
print()
print("  Next: run script 11_export_data.py to update data.json")
