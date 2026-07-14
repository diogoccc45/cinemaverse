"""
SCRIPT 03 - Analysis of enriched data
======================================
Now that we have genres, directors and cast from TMDb,
this script generates 4 charts revealing deeper patterns.
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_FILE  = os.path.join(ROOT_DIR, 'data', 'processed', 'movies_enriched.csv')
OUT_DIR = os.path.join(ROOT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# STYLE
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
GOLD = '#e8c96a'
GREEN = '#00c030'
RED = '#e84040'
BLUE = '#4488ff'
PURPLE = '#9966ff'
TEAL = '#00ccaa'

# LOAD
print("Loading enriched data...")
df = pd.read_csv(DATA_FILE)
print(f"  {len(df)} films loaded.")

# Explode genres, directors, cast into separate rows for analysis
genres_exp = df[df['genres'].notna()].assign(
    genre=df['genres'].str.split('|')
).explode('genre')

directors_exp = df[df['directors'].notna()].assign(
    director=df['directors'].str.split('|')
).explode('director')

cast_exp = df[df['cast'].notna()].assign(
    actor=df['cast'].str.split('|')
).explode('actor')

# CHART 1 - TOP GENRES (count + avg rating)

genre_stats = genres_exp.groupby('genre').agg(
    films=('Name', 'count'),
    avg_rating=('Rating', 'mean')
).query('films >= 10').sort_values('films', ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('My Genre Profile', fontsize=15, color=GOLD)

colors_count = [GREEN if r >= 4.0 else GOLD if r >= 3.5 else TEAL
                for r in genre_stats['avg_rating']]
bars = ax1.barh(genre_stats.index, genre_stats['films'], color=colors_count, zorder=3)
for bar, val in zip(bars, genre_stats['films']):
    ax1.text(val + 1, bar.get_y() + bar.get_height()/2,
             str(val), va='center', fontsize=9, color='#9999b8')
ax1.set_xlabel('Number of films')
ax1.set_title('Films watched per genre')
ax1.set_facecolor('#111118')

colors_rating = [GREEN if r >= 4.0 else GOLD if r >= 3.5 else TEAL
                 for r in genre_stats['avg_rating']]
bars2 = ax2.barh(genre_stats.index, genre_stats['avg_rating'], color=colors_rating, zorder=3)
ax2.axvline(x=df['Rating'].mean(), color=GOLD, linestyle='--',
            linewidth=1, alpha=0.5, label=f'My avg ({df["Rating"].mean():.2f})')
for bar, val in zip(bars2, genre_stats['avg_rating']):
    ax2.text(val + 0.03, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=9, color='#9999b8')
ax2.set_xlabel('Average rating')
ax2.set_title('How I rate each genre')
ax2.set_xlim(0, 5.2)
ax2.set_facecolor('#111118')
ax2.legend(facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_genres.png'), dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/05_genres.png")

# CHART 2 - TOP DIRECTORS (bubble chart, fixed label spacing)

dir_stats = directors_exp.groupby('director').agg(
    films=('Name', 'count'),
    avg_rating=('Rating', 'mean')
).query('films >= 3').sort_values('avg_rating', ascending=False).head(20)

from adjustText import adjust_text

# Taller figure + more padding so labels have room to breathe
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor('#0a0a0f')

scatter = ax.scatter(
    dir_stats['films'],
    dir_stats['avg_rating'],
    s=dir_stats['films'] * 100,
    c=dir_stats['avg_rating'],
    cmap='YlOrRd',
    alpha=0.85,
    zorder=3,
    edgecolors='#333344',
    linewidths=0.5
)

texts = []
for director, row in dir_stats.iterrows():
    texts.append(ax.text(
        row['films'], row['avg_rating'],
        director,
        fontsize=9, color='#ddddee',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e', edgecolor='none', alpha=0.7)
    ))

# More aggressive repulsion so labels move further from bubbles
adjust_text(
    texts,
    ax=ax,
    arrowprops=dict(arrowstyle='-', color='#555577', lw=0.6),
    expand=(2.0, 2.5),
    force_text=(1.0, 1.5),
    force_points=(1.5, 2.0),
)

ax.axhline(y=df['Rating'].mean(), color=GOLD, linestyle='--',
           linewidth=1, alpha=0.5, label=f'My avg ({df["Rating"].mean():.2f})')
ax.set_xlabel('Films watched')
ax.set_ylabel('Average rating I gave')
ax.set_title('My favourite directors  //  bubble size = films watched  //  min 3 films')
ax.set_facecolor('#111118')
ax.legend(facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8')

plt.colorbar(scatter, ax=ax, label='Avg rating').ax.yaxis.label.set_color('#9999b8')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '06_directors.png'), dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/06_directors.png")

# CHART 3 - TOP ACTORS

actor_stats = cast_exp.groupby('actor').agg(
    films=('Name', 'count'),
    avg_rating=('Rating', 'mean')
).query('films >= 5').sort_values('films', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0a0a0f')

colors = [GREEN if r >= 4.0 else GOLD if r >= 3.5 else TEAL if r >= 3.0 else RED
          for r in actor_stats['avg_rating']]

bars = ax.barh(actor_stats.index, actor_stats['films'], color=colors, zorder=3)
for bar, (films, rating) in zip(bars, zip(actor_stats['films'], actor_stats['avg_rating'])):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{films} films  avg {rating:.1f}', va='center', fontsize=8, color='#9999b8')

ax.set_xlabel('Number of films')
ax.set_title('Actors who appear most in my watched films  //  min 5 films')
ax.set_facecolor('#111118')
ax.set_xlim(0, actor_stats['films'].max() + 6)

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=GREEN, label='I rate highly (avg >= 4.0)'),
    Patch(color=GOLD,  label='I like (avg >= 3.5)'),
    Patch(color=TEAL,  label='Mixed feelings (avg >= 3.0)'),
    Patch(color=RED,   label='I tend to dislike'),
], facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '07_actors.png'), dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/07_actors.png")

# CHART 4 - ME VS THE CROWD (contrarian films)
# Two horizontal bar charts: films I loved more + films I hated more
# Much more readable and interesting than a scatter plot

compare = df[(df['Rating'].notna()) & (df['tmdb_rating'] > 0)].copy()
compare['tmdb_rating_5'] = compare['tmdb_rating'] / 2
compare['diff'] = compare['Rating'] - compare['tmdb_rating_5']

# Top contrarian films in each direction
loved = compare[compare['diff'] > 0].nlargest(10, 'diff')
hated = compare[compare['diff'] < 0].nsmallest(10, 'diff')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle(
    f'Me vs The Crowd  //  my avg bias: {compare["diff"].mean():+.2f}  //  '
    f'I rate higher on {(compare["diff"] > 0).sum()} films, lower on {(compare["diff"] < 0).sum()}',
    fontsize=13, color=GOLD
)

# Left: films I loved more than the crowd
loved_sorted = loved.sort_values('diff')
bars1 = ax1.barh(loved_sorted['Name'], loved_sorted['diff'], color=GREEN, zorder=3)
for bar, (_, row) in zip(bars1, loved_sorted.iterrows()):
    ax1.text(
        bar.get_width() + 0.03,
        bar.get_y() + bar.get_height() / 2,
        f'me {row["Rating"]:.1f}  crowd {row["tmdb_rating_5"]:.1f}',
        va='center', fontsize=8, color='#9999b8'
    )
ax1.set_xlabel('My rating minus crowd rating')
ax1.set_title('Films I loved MORE than the crowd')
ax1.set_facecolor('#111118')
ax1.set_xlim(0, loved_sorted['diff'].max() + 0.8)

# Right: films I hated more than the crowd
hated_sorted = hated.sort_values('diff', ascending=False)
bars2 = ax2.barh(hated_sorted['Name'], hated_sorted['diff'].abs(), color=RED, zorder=3)
for bar, (_, row) in zip(bars2, hated_sorted.iterrows()):
    ax2.text(
        bar.get_width() + 0.03,
        bar.get_y() + bar.get_height() / 2,
        f'me {row["Rating"]:.1f}  crowd {row["tmdb_rating_5"]:.1f}',
        va='center', fontsize=8, color='#9999b8'
    )
ax2.set_xlabel('Crowd rating minus my rating')
ax2.set_title('Films I hated MORE than the crowd')
ax2.set_facecolor('#111118')
ax2.set_xlim(0, hated_sorted['diff'].abs().max() + 0.8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '08_me_vs_crowd.png'), dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/08_me_vs_crowd.png")

# SUMMARY
total_hours = df[df['runtime'] > 0]['runtime'].sum() / 60
top_genre    = genres_exp.groupby('genre').size().idxmax()
top_director = directors_exp.groupby('director').size().idxmax()
top_actor    = cast_exp.groupby('actor').size().idxmax()

print()
print("=" * 50)
print("  DONE! 4 charts saved in output/")
print("=" * 50)
print(f"  Total hours watched:   {total_hours:.0f}h ({total_hours/24:.0f} days!)")
print(f"  Most watched genre:    {top_genre}")
print(f"  Most watched director: {top_director}")
print(f"  Most watched actor:    {top_actor}")
print()