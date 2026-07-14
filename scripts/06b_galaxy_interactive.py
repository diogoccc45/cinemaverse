"""
SCRIPT 06b - Interactive Film Galaxy (Plotly)
=============================================
Generates an interactive HTML file with the film galaxy.
Hover over any point to see film details.
Three tabs: by cluster, by rating, by decade.
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')
EMBED_FILE = os.path.join(ROOT_DIR, 'data', 'processed', 'embeddings.csv')
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'processed', 'movies_enriched.csv')
OUT_DIR = os.path.join(ROOT_DIR, 'output')

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────

emb = pd.read_csv(EMBED_FILE)
extra = pd.read_csv(DATA_FILE)

# Only bring columns not already in embeddings.csv
df = emb.merge(
    extra[['Name', 'Year', 'countries', 'runtime', 'tmdb_rating', 'overview']],
    on=['Name', 'Year'], how='left'
)

df['decade'] = (df['Year'] // 10 * 10).astype(int).astype(str) + 's'
df['stars']  = df['Rating'].apply(lambda r: '★' * int(r) + ('½' if r % 1 == 0.5 else '') if pd.notna(r) else '?')

# Hover text — shown when you mouse over a point
df['hover'] = df.apply(lambda row: (
    f"<b>{row['Name']}</b> ({int(row['Year'])})<br>"
    f"My rating: {row['stars']} ({row['Rating']})<br>"
    f"Director: {str(row['directors']).split('|')[0] if pd.notna(row['directors']) else 'unknown'}<br>"
    f"Genres: {str(row['genres']).replace('|', ', ') if pd.notna(row['genres']) else 'unknown'}<br>"
    f"TMDb: {row['tmdb_rating']:.1f}/10<br>"
    f"Runtime: {int(row['runtime'])} min" if pd.notna(row['runtime']) and row['runtime'] > 0
    else f"<b>{row['Name']}</b> ({int(row['Year'])})<br>My rating: {row['stars']}"
), axis=1)

print(f"  {len(df)} films loaded.")

# ─────────────────────────────────────────────────────────────
# BUILD INTERACTIVE CHARTS
# ─────────────────────────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

BG = '#0a0a0f'
BG2 = '#111118'
GOLD = '#e8c96a'
MUTED = '#666680'
TEXT = '#e8e8f0'

# ── FIGURE 1: by cluster ──

fig1 = go.Figure()

cluster_ids = sorted(df['cluster'].unique())
# Use a rich colour palette
palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

for i, cid in enumerate(cluster_ids):
    mask = df['cluster'] == cid
    cdf  = df[mask]

    if cid == -1:
        color = '#333344'
        name  = 'Noise (no cluster)'
    else:
        color = palette[i % len(palette)]
        top_g = cdf['genres'].str.split('|').explode().value_counts().head(2).index.tolist()
        name  = f"#{cid} — {' · '.join(top_g)} ({mask.sum()} films)"

    fig1.add_trace(go.Scatter(
        x=cdf['umap_x'],
        y=cdf['umap_y'],
        mode='markers',
        name=name,
        text=cdf['hover'],
        hovertemplate='%{text}<extra></extra>',
        marker=dict(
            size=cdf['Rating'].fillna(2.5) * 3.5,
            color=color,
            opacity=0.8 if cid != -1 else 0.25,
            line=dict(width=0),
        )
    ))

fig1.update_layout(
    title=dict(
        text='My Film Galaxy — by Taste Cluster',
        font=dict(color=GOLD, size=18, family='monospace'),
        x=0.5
    ),
    paper_bgcolor=BG,
    plot_bgcolor=BG2,
    font=dict(color=TEXT, family='monospace'),
    xaxis=dict(title='UMAP dimension 1', gridcolor='#1e1e2e', zerolinecolor='#333344'),
    yaxis=dict(title='UMAP dimension 2', gridcolor='#1e1e2e', zerolinecolor='#333344'),
    legend=dict(
        bgcolor='#1a1a24', bordercolor='#333344', borderwidth=1,
        font=dict(size=10), itemsizing='constant'
    ),
    hoverlabel=dict(bgcolor='#1a1a24', bordercolor=GOLD, font=dict(color=TEXT, size=12)),
    height=750,
)

# ── FIGURE 2: by rating ──

fig2 = go.Figure()

# One trace per rating value for clean legend
rating_values = sorted(df['Rating'].dropna().unique())
rating_cmap = {
    0.5: '#e84040', 1.0: '#e85a40', 1.5: '#e87040',
    2.0: '#e89040', 2.5: '#e8b040', 3.0: '#e8c96a',
    3.5: '#c8e060', 4.0: '#80cc40', 4.5: '#40b830', 5.0: '#00c030'
}

for rating in rating_values:
    mask = df['Rating'] == rating
    cdf  = df[mask]
    fig2.add_trace(go.Scatter(
        x=cdf['umap_x'],
        y=cdf['umap_y'],
        mode='markers',
        name=f'{rating}★  ({mask.sum()} films)',
        text=cdf['hover'],
        hovertemplate='%{text}<extra></extra>',
        marker=dict(
            size=rating * 3.5,
            color=rating_cmap.get(rating, GOLD),
            opacity=0.85,
            line=dict(width=0),
        )
    ))

fig2.update_layout(
    title=dict(
        text='My Film Galaxy — by My Rating',
        font=dict(color=GOLD, size=18, family='monospace'),
        x=0.5
    ),
    paper_bgcolor=BG,
    plot_bgcolor=BG2,
    font=dict(color=TEXT, family='monospace'),
    xaxis=dict(title='UMAP dimension 1', gridcolor='#1e1e2e', zerolinecolor='#333344'),
    yaxis=dict(title='UMAP dimension 2', gridcolor='#1e1e2e', zerolinecolor='#333344'),
    legend=dict(
        bgcolor='#1a1a24', bordercolor='#333344', borderwidth=1,
        font=dict(size=10), itemsizing='constant',
        title=dict(text='My rating', font=dict(color=GOLD))
    ),
    hoverlabel=dict(bgcolor='#1a1a24', bordercolor=GOLD, font=dict(color=TEXT, size=12)),
    height=750,
)

# ── FIGURE 3: by decade ──

fig3 = go.Figure()

decade_colors = {
    '1920s': '#9b59b6', '1930s': '#8e44ad', '1950s': '#2471a3',
    '1960s': '#1a5276', '1970s': '#16a085', '1980s': '#1e8449',
    '1990s': '#f39c12', '2000s': '#e67e22', '2010s': '#e8c96a',
    '2020s': '#00c030'
}

for decade in sorted(df['decade'].unique()):
    mask = df['decade'] == decade
    cdf  = df[mask]
    fig3.add_trace(go.Scatter(
        x=cdf['umap_x'],
        y=cdf['umap_y'],
        mode='markers',
        name=f'{decade}  ({mask.sum()} films)',
        text=cdf['hover'],
        hovertemplate='%{text}<extra></extra>',
        marker=dict(
            size=cdf['Rating'].fillna(2.5) * 3.5,
            color=decade_colors.get(decade, MUTED),
            opacity=0.85,
            line=dict(width=0),
        )
    ))

fig3.update_layout(
    title=dict(
        text='My Film Galaxy — by Decade',
        font=dict(color=GOLD, size=18, family='monospace'),
        x=0.5
    ),
    paper_bgcolor=BG,
    plot_bgcolor=BG2,
    font=dict(color=TEXT, family='monospace'),
    xaxis=dict(title='UMAP dimension 1', gridcolor='#1e1e2e', zerolinecolor='#333344'),
    yaxis=dict(title='UMAP dimension 2', gridcolor='#1e1e2e', zerolinecolor='#333344'),
    legend=dict(
        bgcolor='#1a1a24', bordercolor='#333344', borderwidth=1,
        font=dict(size=10), itemsizing='constant',
        title=dict(text='Decade', font=dict(color=GOLD))
    ),
    hoverlabel=dict(bgcolor='#1a1a24', bordercolor=GOLD, font=dict(color=TEXT, size=12)),
    height=750,
)

# ─────────────────────────────────────────────────────────────
# COMBINE INTO ONE HTML FILE WITH TABS
# ─────────────────────────────────────────────────────────────

html1 = fig1.to_html(full_html=False, include_plotlyjs=True)
html2 = fig2.to_html(full_html=False, include_plotlyjs=False)
html3 = fig3.to_html(full_html=False, include_plotlyjs=False)

html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>My Film Galaxy — diogocc</title>

  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0a0a0f; color: #e8e8f0; font-family: 'Courier New', monospace; }}
    header {{
      padding: 32px 40px 20px;
      border-bottom: 1px solid #1e1e2e;
    }}
    header h1 {{
      font-size: 28px; color: #e8c96a; letter-spacing: -0.02em;
    }}
    header p {{
      font-size: 12px; color: #666680; margin-top: 6px;
    }}
    .tabs {{
      display: flex; gap: 0;
      padding: 0 40px;
      border-bottom: 1px solid #1e1e2e;
    }}
    .tab {{
      padding: 14px 24px;
      font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase;
      color: #666680; cursor: pointer;
      border-bottom: 2px solid transparent;
      transition: color 0.2s, border-color 0.2s;
    }}
    .tab:hover {{ color: #e8c96a; }}
    .tab.active {{ color: #e8c96a; border-color: #e8c96a; }}
    .panel {{ display: none; padding: 24px 40px; }}
    .panel.active {{ display: block; }}
    .hint {{
      font-size: 11px; color: #444460; margin-bottom: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>My Film Galaxy</h1>
    <p>768 films · UMAP projection of genres, rating, director · HDBSCAN clusters · hover any point for details</p>
  </header>

  <div class="tabs">
    <div class="tab active" onclick="showTab(0)">By Cluster</div>
    <div class="tab" onclick="showTab(1)">By Rating</div>
    <div class="tab" onclick="showTab(2)">By Decade</div>
  </div>

  <div class="panel active" id="panel0">
    <p class="hint">Hover over any film to see its details. Click legend items to hide/show clusters.</p>
    {html1}
  </div>
  <div class="panel" id="panel1">
    <p class="hint">Larger points = higher rating. Green = loved it, Red = hated it.</p>
    {html2}
  </div>
  <div class="panel" id="panel2">
    <p class="hint">Decades are well mixed — genres matter more than era in my taste.</p>
    {html3}
  </div>

  <script>
    function showTab(idx) {{
      document.querySelectorAll('.tab').forEach((t, i) => {{
        t.classList.toggle('active', i === idx);
      }});
      document.querySelectorAll('.panel').forEach((p, i) => {{
        p.classList.toggle('active', i === idx);
      }});
    }}
  </script>
</body>
</html>"""

out_path = os.path.join(OUT_DIR, 'film_galaxy_interactive.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html_out)

print(f"  Saved: output/film_galaxy_interactive.html")