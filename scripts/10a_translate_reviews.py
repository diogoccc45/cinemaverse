"""
SCRIPT 10a - Translate reviews to English
==========================================
Translates all Portuguese reviews to English using
Google Translate (free, via deep-translator).

Saves progress so if it stops halfway, it continues.
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import json
import time
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, '..')
RAW_DIR    = os.path.join(ROOT_DIR, 'data', 'raw')
DATA_DIR   = os.path.join(ROOT_DIR, 'data', 'processed')

CACHE_FILE = os.path.join(DATA_DIR, 'translation_cache.json')
OUTPUT_FILE = os.path.join(DATA_DIR, 'reviews_translated.csv')

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
print("Loading reviews...")

reviews = pd.read_csv(os.path.join(RAW_DIR, 'reviews.csv'))
reviews = reviews[reviews['Review'].notna()].copy().reset_index(drop=True)
print(f"  {len(reviews)} reviews to process")

# Load translation cache
cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    print(f"  Cache loaded: {len(cache)} already translated")

# ─────────────────────────────────────────────────────────────
# DETECT LANGUAGE
# ─────────────────────────────────────────────────────────────
from langdetect import detect, LangDetectException

def detect_language(text):
    try:
        return detect(str(text))
    except LangDetectException:
        return 'en'  # assume English if detection fails

print("\nDetecting languages...")
reviews['lang'] = reviews['Review'].apply(detect_language)

lang_counts = reviews['lang'].value_counts()
print(f"  Language distribution:")
for lang, count in lang_counts.items():
    print(f"    {lang}: {count} reviews")

# ─────────────────────────────────────────────────────────────
# TRANSLATE
# ─────────────────────────────────────────────────────────────
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='auto', target='en')

def translate(text):
    """Translate text to English. Returns original if already English."""
    text = str(text).strip()
    if not text:
        return text

    # Check cache first
    if text in cache:
        return cache[text]

    try:
        # Google Translate has a 5000 char limit per request
        if len(text) > 4500:
            # Split into chunks and translate each
            chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
            translated = ' '.join(translator.translate(chunk) for chunk in chunks)
        else:
            translated = translator.translate(text)

        cache[text] = translated
        return translated
    except Exception as e:
        print(f"    Translation error: {e}")
        return text  # return original if translation fails

print("\nTranslating reviews...")
translated_reviews = []

for i, row in reviews.iterrows():
    lang = row['lang']
    text = row['Review']

    if lang == 'en':
        # Already English — keep as is
        translated = text
        status = 'kept'
    else:
        print(f"  [{i+1}/{len(reviews)}] {row['Name'][:35]} ({lang})", end='', flush=True)
        translated = translate(text)
        status = 'translated'
        print(f" -> done")
        time.sleep(0.3)  # be polite to Google

    translated_reviews.append({
        'Name':              row['Name'],
        'Year':              row['Year'],
        'Rating':            row['Rating'],
        'original_review':   text,
        'translated_review': translated,
        'original_lang':     lang,
        'status':            status,
    })

    # Save cache every 50 reviews
    if (i + 1) % 50 == 0:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"  -- Cache saved ({i+1} done) --")

# Final cache save
with open(CACHE_FILE, 'w', encoding='utf-8') as f:
    json.dump(cache, f, ensure_ascii=False)

# Save output
df_out = pd.DataFrame(translated_reviews)
df_out.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
translated_count = (df_out['status'] == 'translated').sum()
kept_count       = (df_out['status'] == 'kept').sum()

print("=" * 50)
print(f"  Total reviews:   {len(df_out)}")
print(f"  Translated:      {translated_count}")
print(f"  Already English: {kept_count}")
print(f"  Saved to: data/processed/reviews_translated.csv")
print()
print("  Sample translated review:")
sample = df_out[df_out['status'] == 'translated'].iloc[0]
print(f"  Film: {sample['Name']}")
print(f"  Original ({sample['original_lang']}): {sample['original_review'][:100]}...")
print(f"  Translated: {sample['translated_review'][:100]}...")
