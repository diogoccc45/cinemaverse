"""
SCRIPT 02b - Fetch TMDb metadata for watchlist
===============================================
Same pipeline as 02, but runs over watchlist.csv instead of ratings.csv.
Fetches genres, director, cast, country, runtime for all 248 watchlist films.

Results are saved to data/processed/watchlist_enriched.csv

Progress is cached in data/processed/watchlist_cache.json.
"""

import sys
import io

# Force UTF-8 output on Windows - fixes encoding errors with non-ASCII names
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import requests
import json
import time
import os

# PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')
RAW_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
OUT_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_FILE = os.path.join(OUT_DIR, 'watchlist_cache.json')
OUTPUT_FILE = os.path.join(OUT_DIR, 'watchlist_enriched.csv')

# API KEY - reads from .env file
env_path = os.path.join(ROOT_DIR, '.env')
API_KEY = None

if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('TMDB_KEY='):
                API_KEY = line.split('=', 1)[1].strip()
                break

if not API_KEY:
    # Fallback: ask the user to type it
    API_KEY = input("Paste your TMDb API key: ").strip()

BASE_URL = 'https://api.themoviedb.org/3'

# API FUNCTIONS

def api_get(endpoint, params={}):
    # Makes a GET request to the TMDb API.
    # Returns the JSON response as a Python dict, or None on error.

    params = dict(params)
    params['api_key'] = API_KEY

    try:
        response = requests.get(
            f'{BASE_URL}{endpoint}',
            params=params,
            timeout=10
        )
        response.raise_for_status()  # raises error if status != 200
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"    API error: {e}")
        return None


def search_movie(title, year):
    """
    Searches TMDb for a film by title + year.
    Returns the TMDb ID (integer) or None if not found.

    Why two attempts?
    Some films aren't found with year filter (e.g. release date
    differences between countries), so we retry without year.
    """
    # Attempt 1: with year filter
    result = api_get('/search/movie', {'query': title, 'year': year, 'language': 'en-US'})
    if result and result.get('results'):
        return result['results'][0]['id']

    # Attempt 2: without year filter (broader search)
    result = api_get('/search/movie', {'query': title, 'language': 'en-US'})
    if result and result.get('results'):
        return result['results'][0]['id']

    return None


def get_movie_details(tmdb_id):
    """
    Fetches full details for a film using its TMDb ID.

    append_to_response=credits means we get cast + crew in the
    same API call instead of making a separate request.
    This halves the number of API calls needed. I hope.
    """
    result = api_get(f'/movie/{tmdb_id}', {
        'append_to_response': 'credits',
        'language': 'en-US'
    })

    if not result:
        return {}

    # Extract genres as a pipe-separated string
    # e.g. "Drama|Crime|Thriller"
    genres = '|'.join(g['name'] for g in result.get('genres', []))

    # Extract director(s) from the crew list
    # crew is a list of people, each with a 'job' field
    directors = '|'.join(
        p['name'] for p in result.get('credits', {}).get('crew', [])
        if p['job'] == 'Director'
    )

    # Extract top 5 cast members
    cast = '|'.join(
        p['name'] for p in result.get('credits', {}).get('cast', [])[:5]
    )

    # Extract production countries as ISO codes
    # e.g. "US|GB|FR"
    countries = '|'.join(
        c['iso_3166_1'] for c in result.get('production_countries', [])
    )

    return {
        'tmdb_id':     tmdb_id,
        'genres':      genres,
        'directors':   directors,
        'cast':        cast,
        'countries':   countries,
        'runtime':     result.get('runtime', 0),
        'budget':      result.get('budget', 0),
        'revenue':     result.get('revenue', 0),
        'tmdb_rating': round(result.get('vote_average', 0), 1),
        'tmdb_votes':  result.get('vote_count', 0),
        'overview':    result.get('overview', ''),
        'tagline':     result.get('tagline', ''),
    }


# MAIN PIPELINE
def main():
    # Load cache (progress from previous run, if any)
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"Cache loaded: {len(cache)} films already fetched.")
    else:
        print("No cache found - starting fresh.")

    # Load watchlist
    ratings = pd.read_csv(os.path.join(RAW_DIR, 'watchlist.csv'))
    total = len(ratings)
    print(f"Watchlist films to process: {total}")
    print(f"Remaining:                  {total - len(cache)}")
    print()

    results = []
    not_found = []

    for i, row in ratings.iterrows():
        title = row['Name']
        year_raw = row.get('Year', 2000)
        year = int(year_raw) if year_raw and str(year_raw) != 'nan' else 2000
        key   = f"{title}_{year}"  # unique cache key

        # Already in cache? Skip the API call
        if key in cache:
            results.append({**row.to_dict(), **cache[key]})
            continue

        print(f"[{i+1}/{total}] {title} ({year})", end='', flush=True)

        # Step 1: get TMDb ID
        tmdb_id = search_movie(title, year)
        time.sleep(0.15)  # be polite to the API, pleaaaaaaassssse

        if not tmdb_id:
            print(" - NOT FOUND")
            not_found.append(f"{title} ({year})")
            cache[key] = {}
            results.append({**row.to_dict()})
            continue

        # Step 2: get full details
        details = get_movie_details(tmdb_id)
        time.sleep(0.15)

        director_str = details.get('directors', 'unknown')
        genres_str   = details.get('genres', 'unknown')
        print(f" - {director_str} | {genres_str[:35]}")

        cache[key] = details
        results.append({**row.to_dict(), **details})

        # Save cache every 50 films (safety net)
        if (i + 1) % 50 == 0:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False)
            print(f"\n  -- Cache saved ({i+1} films done) --\n")

    # Final cache save
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)

    # Save enriched CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

    # Summary
    print()
    print("=" * 50)
    print(f"  DONE!")
    print(f"  {len(results)} films saved to data/processed/watchlist_enriched.csv")
    if not_found:
        print(f"  {len(not_found)} films not found on TMDb:")
        for f in not_found:
            print(f"    - {f}")
    print("=" * 50)
    print()

if __name__ == '__main__':
    main()