"""
SCRIPT 02 - Fetch TMDb metadata for all films
==============================================
For each film in ratings.csv, this script:
  1. Searches TMDb by title + year to get the TMDb ID
  2. Fetches full details: genres, director, cast, country, runtime

Results are saved to data/processed/movies_enriched.csv

Progress is cached in data/processed/tmdb_cache.json so if the script stops halfway, it picks up where it left off.
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

CACHE_FILE = os.path.join(OUT_DIR, 'tmdb_cache.json')
OUTPUT_FILE = os.path.join(OUT_DIR, 'movies_enriched.csv')

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
    Searches TMDb for a title by title + year.
    Tries movie first, then TV (for specials, series, etc.)
    Returns (tmdb_id, media_type) or (None, None) if not found.
    """
    # Attempt 1: movie with year
    result = api_get('/search/movie', {'query': title, 'year': year, 'language': 'en-US'})
    if result and result.get('results'):
        return result['results'][0]['id'], 'movie'

    # Attempt 2: movie without year
    result = api_get('/search/movie', {'query': title, 'language': 'en-US'})
    if result and result.get('results'):
        return result['results'][0]['id'], 'movie'

    # Attempt 3: TV (specials, series, stand-up, documentaries)
    result = api_get('/search/tv', {'query': title, 'first_air_date_year': year, 'language': 'en-US'})
    if result and result.get('results'):
        return result['results'][0]['id'], 'tv'

    # Attempt 4: TV without year
    result = api_get('/search/tv', {'query': title, 'language': 'en-US'})
    if result and result.get('results'):
        return result['results'][0]['id'], 'tv'

    return None, None


def get_movie_details(tmdb_id, media_type='movie'):
    """
    Fetches full details for a film or TV show using its TMDb ID.
    Handles both /movie/ and /tv/ endpoints.
    """
    endpoint = f'/{media_type}/{tmdb_id}'
    result = api_get(endpoint, {
        'append_to_response': 'credits',
        'language': 'en-US'
    })

    if not result:
        return {}

    # Extract genres as a pipe-separated string
    # e.g. "Drama|Crime|Thriller"
    genres = '|'.join(g['name'] for g in result.get('genres', []))

    # Extract director(s) from crew — works for both movie and TV
    crew = result.get('credits', {}).get('crew', [])
    directors = '|'.join(
        p['name'] for p in crew if p.get('job') == 'Director'
    )
    # For TV, also check created_by if no directors found
    if not directors and result.get('created_by'):
        directors = '|'.join(p['name'] for p in result['created_by'])

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
        'runtime':     result.get('runtime') or (result.get('episode_run_time', [0])[0] if result.get('episode_run_time') else 0),
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
        # Remove empty cache entries so missing films get retried
        cache = {k: v for k, v in cache.items() if v}
        print(f"Cache loaded: {len(cache)} films already fetched.")
    else:
        print("No cache found - starting fresh.")

    # Load ratings (our list of films to enrich)
    ratings = pd.read_csv(os.path.join(RAW_DIR, 'ratings.csv'))
    total = len(ratings)
    print(f"Films to process: {total}")
    print(f"Remaining:        {total - len(cache)}")
    print()

    results = []
    not_found = []

    for i, row in ratings.iterrows():
        title = row['Name']
        year  = int(row['Year'])
        key   = f"{title}_{year}"  # unique cache key

        # Already in cache? Skip the API call
        if key in cache:
            results.append({**row.to_dict(), **cache[key]})
            continue

        print(f"[{i+1}/{total}] {title} ({year})", end='', flush=True)

        # Step 1: get TMDb ID
        tmdb_id, media_type = search_movie(title, year)
        time.sleep(0.15)  # be polite to the API, pleaaaaaaassssse

        if not tmdb_id:
            print(" - NOT FOUND")
            not_found.append(f"{title} ({year})")
            cache[key] = {}
            results.append({**row.to_dict()})
            continue

        # Step 2: get full details
        details = get_movie_details(tmdb_id, media_type)
        time.sleep(0.15)

        director_str = details.get('directors', 'unknown')
        genres_str = details.get('genres', 'unknown')
        print(f" [{media_type}] - {director_str} | {genres_str[:35]}")

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
    print(f"  {len(results)} films saved to data/processed/movies_enriched.csv")
    if not_found:
        print(f"  {len(not_found)} films not found on TMDb:")
        for f in not_found:
            print(f"    - {f}")
    print("=" * 50)
    print()

if __name__ == '__main__':
    main()