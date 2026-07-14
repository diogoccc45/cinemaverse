"""
SCRIPT 11c - Fetch director photos and bios from TMDb
======================================================
For each director with >= 3 films in your history,
fetches their photo, biography, birthday and known_for.

How to run:
    python scripts/11c_fetch_directors.py
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

CACHE_FILE = os.path.join(DATA_DIR, 'directors_cache.json')
OUTPUT_FILE = os.path.join(DATA_DIR, 'directors_enriched.json')
TMDB_IMG = 'https://image.tmdb.org/t/p/w300'

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
    print(f"Cache loaded: {len(cache)} directors")

# Load enriched data and find top directors
enriched = pd.read_csv(os.path.join(DATA_DIR, 'movies_enriched.csv'))
dirs_exp = enriched[enriched['directors'].notna()].assign(
    director=enriched['directors'].str.split('|')
).explode('director')

dir_stats = dirs_exp.groupby('director').agg(
    count=('Name', 'count'),
    avg_rating=('Rating', 'mean'),
    films=('Name', list)
).query('count >= 3').reset_index()

print(f"Directors to fetch: {len(dir_stats)}")

results = {}

for i, row in dir_stats.iterrows():
    name = row['director']

    if name in cache:
        results[name] = cache[name]
        continue

    print(f"[{i+1}/{len(dir_stats)}] {name}", end='', flush=True)

    # Search for person
    search = api_get('/search/person', {'query': name, 'language': 'en-US'})
    time.sleep(0.15)

    if not search or not search.get('results'):
        print(" - not found")
        results[name] = {'name': name}
        cache[name]   = {'name': name}
        continue

    # Find the most relevant result (director)
    person = None
    for p in search['results']:
        if p.get('known_for_department') == 'Directing':
            person = p
            break
    if not person:
        person = search['results'][0]

    person_id = person['id']

    # Get full details
    details = api_get(f'/person/{person_id}', {'language': 'en-US'})
    time.sleep(0.15)

    if not details:
        print(" - no details")
        results[name] = {'name': name}
        cache[name]   = {'name': name}
        continue

    photo = f"{TMDB_IMG}{details['profile_path']}" if details.get('profile_path') else None
    bio   = details.get('biography', '') if details.get('biography') else None

    entry = {
        'name':       name,
        'tmdb_id':    person_id,
        'photo':      photo,
        'bio':        bio,
        'birthday':   details.get('birthday'),
        'birthplace': details.get('place_of_birth'),
        'count':      int(row['count']),
        'avg_rating': round(float(row['avg_rating']), 2),
    }

    print(f" -> {'📷' if photo else '—'}  {bio[:50] + '...' if bio else 'no bio'}")
    results[name] = entry
    cache[name]   = entry

    if (i + 1) % 20 == 0:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"\n  -- Cache saved ({i+1} done) --\n")

# Final save
with open(CACHE_FILE, 'w', encoding='utf-8') as f:
    json.dump(cache, f, ensure_ascii=False)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

has_photo = sum(1 for v in results.values() if v.get('photo'))
has_bio   = sum(1 for v in results.values() if v.get('bio'))

print()
print("=" * 50)
print("  DONE!")
print("=" * 50)
print(f"  {len(results)} directors fetched")
print(f"  {has_photo} with photo")
print(f"  {has_bio} with biography")
print(f"  Saved to data/processed/directors_enriched.json")
