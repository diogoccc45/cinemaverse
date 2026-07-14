# 🎬 Cinemaverse

A personal data science project built from my [Letterboxd](https://letterboxd.com/diogocc) watch history — combining network science, machine learning and NLP to analyse, visualise and predict my film taste.

## What it does

**Exploration** — genre profiles, director bubbles, actor rankings, production company analysis, me vs the crowd

**Network Science** — film graph built from shared actors/directors, Louvain community detection, comparison with Erdős–Rényi, Barabási–Albert and Watts–Strogatz models

**Galaxy** — every film projected into 2D space via UMAP + HDBSCAN clustering, coloured by rating, cluster or decade

**Recommendations** — cosine similarity and semantic embeddings (sentence-transformers) to rank my watchlist and discover new films

**Rating Prediction** — Random Forest trained on my ratings, with weighted director/genre/franchise features

**Director Deep-Dive** — per-director profile with photo, biography, film grid, rating timeline, word cloud of my reviews and head-to-head comparison

**Reviews** — all my reviews filterable by rating and text, with a DistilGPT-2 model fine-tuned on my writing style to generate new ones

**Lists** — my curated Letterboxd lists with TMDb poster art

## Stack

| Layer | Tools |
|---|---|
| Data | Letterboxd export, TMDb API |
| Analysis | pandas, NetworkX, UMAP, HDBSCAN |
| ML | scikit-learn, sentence-transformers, transformers (DistilGPT-2) |
| Visualisation | Plotly, matplotlib, wordcloud |
| Dashboard | Streamlit |
| Model hosting | Hugging Face Hub (`diogocc45/letterboxd-gpt2`) |

## Structure

```
cinemaverse/
├── scripts/
│   ├── 01_explore.py
│   ├── 02_tmdb_fetch.py
│   ├── 02b_tmdb_fetch.py
│   ├── 03_analysis.py
│   ├── 04_advanced.py
│   ├── 05_network_science.py
│   ├── 06_embeddings_clustering.py
│   ├── 06b_galaxy_interactive.py
│   ├── 07_network_comparison.py
│   ├── 08a_fetch_candidates.py
│   ├── 08b_recommendations.py
│   ├── 09b_semantic_recommendations.py
│   ├── 10a_translate_reviews.py
│   ├── 11_export_data.py
│   ├── 11b_fetch_posters.py
│   ├── 11c_fetch_directors.py
│   └── 12_upload_model.py
├── notebooks/
│   ├── 09_rating_prediction.ipynb
│   └── 10b_finetune_gpt2.ipynb
├── app.py                # Streamlit dashboard
├── data.json             # all processed data for the dashboard
├── requirements.txt
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

[diogocc](https://letterboxd.com/diogocc) · built with Claude Sonnet 4.6 Medium Effort.