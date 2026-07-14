"""
SCRIPT 01 - Letterboxd Dataset Exploration
===========================================
Generates 4 charts saved as PNG files in a folder called "output/".

How to run:
    pip install pandas matplotlib
    python scripts/01_explore.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


# PATHS
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, '..')
RAW_DIR    = os.path.join(ROOT_DIR, 'data', 'raw')
OUT_DIR    = os.path.join(ROOT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# STYLE - cinema dark theme😎
plt.rcParams.update({
    'figure.facecolor':   '#0a0a0f',
    'axes.facecolor':     '#111118',
    'axes.edgecolor':     '#333344',
    'axes.labelcolor':    '#9999b8',
    'axes.titlecolor':    '#e8e8f0',
    'axes.titlesize':     13,
    'axes.titlepad':      14,
    'axes.grid':          True,
    'grid.color':         '#1e1e2e',
    'grid.linewidth':     0.6,
    'xtick.color':        '#666680',
    'ytick.color':        '#666680',
    'text.color':         '#e8e8f0',
    'font.family':        'monospace',
})

GOLD   = '#e8c96a'
GREEN  = '#00c030'
RED    = '#e84040'
MUTED  = '#444460'
BLUE   = '#4488ff'
PURPLE = '#9966ff'

# LOAD DATA
ratings   = pd.read_csv(os.path.join(RAW_DIR, 'ratings.csv'))
diary     = pd.read_csv(os.path.join(RAW_DIR, 'diary.csv'))
reviews   = pd.read_csv(os.path.join(RAW_DIR, 'reviews.csv'))
watched   = pd.read_csv(os.path.join(RAW_DIR, 'watched.csv'))
watchlist = pd.read_csv(os.path.join(RAW_DIR, 'watchlist.csv'))
likes     = pd.read_csv(os.path.join(RAW_DIR, 'likes', 'films.csv'))

diary['Date']   = pd.to_datetime(diary['Date'])
ratings['Date'] = pd.to_datetime(ratings['Date'])
reviews['Date'] = pd.to_datetime(reviews['Date'])

ratings['decade']   = (ratings['Year'] // 10 * 10).astype(int)
diary['yearmonth']  = diary['Date'].dt.to_period('M')

reviews_with_text = reviews[reviews['Review'].notna()].copy()
reviews_with_text['length'] = reviews_with_text['Review'].str.len()

print(f"  {len(ratings)} rated films")
print(f"  {len(diary)} diary entries")
print(f"  {len(reviews_with_text)} reviews")

# CHART 1 - RATING DISTRIBUTION

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0a0a0f')

dist = ratings['Rating'].value_counts().sort_index()
colors = []
for r in dist.index:
    if r <= 1.5:   colors.append(RED)
    elif r <= 3.0: colors.append(MUTED)
    elif r <= 4.0: colors.append(GOLD)
    else:          colors.append(GREEN)

bars = ax.bar([str(r) for r in dist.index], dist.values, color=colors, width=0.7, zorder=3)

for bar, val in zip(bars, dist.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            str(val), ha='center', va='bottom', fontsize=9, color='#9999b8')

ax.set_xlabel('Rating')
ax.set_ylabel('Number of films')
ax.set_title('Rating Distribution  //  767 films  //  avg 3.32')
ax.set_facecolor('#111118')

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=RED,   label='Bad (<=1.5)'),
    Patch(color=MUTED, label='OK (2-3)'),
    Patch(color=GOLD,  label='Good (3.5-4)'),
    Patch(color=GREEN, label='Masterpiece (4.5-5)'),
], loc='upper left', facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_rating_distribution.png'), dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/01_rating_distribution.png")

# CHART 2 - ACTIVITY OVER TIME

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[2, 1])
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('Viewing Activity Over Time')

monthly = diary.groupby('yearmonth').size().reset_index(name='count')
monthly['label'] = monthly['yearmonth'].astype(str)

bar_colors = [GREEN if c >= 40 else GOLD if c >= 20 else '#2a2a4a' for c in monthly['count']]
ax1.bar(range(len(monthly)), monthly['count'], color=bar_colors, zorder=3)
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly['label'], rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Films per month')
ax1.set_title('Monthly activity  //  peak: Apr 2025 (46 films)')
ax1.set_facecolor('#111118')

peak_idx = monthly['count'].idxmax()
ax1.annotate(f"  {monthly.loc[peak_idx, 'count']} films",
             xy=(peak_idx, monthly.loc[peak_idx, 'count']), color=GREEN, fontsize=9)

ax2.plot(range(len(monthly)), monthly['count'].cumsum(), color=GOLD, linewidth=2, zorder=3)
ax2.fill_between(range(len(monthly)), monthly['count'].cumsum(), alpha=0.15, color=GOLD)
ax2.set_xticks(range(len(monthly)))
ax2.set_xticklabels(monthly['label'], rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Total films')
ax2.set_title('Cumulative films watched')
ax2.set_facecolor('#111118')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_activity_over_time.png'), dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/02_activity_over_time.png")

# CHART 3 - DECADES

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('Films by Decade')

decade_data = ratings.groupby('decade').agg(
    count=('Name', 'count'),
    avg_rating=('Rating', 'mean')
).reset_index()
decade_labels = [str(int(d)) + 's' for d in decade_data['decade']]
dec_colors = [PURPLE] * 2 + [BLUE] * 2 + ['#16a085'] + ['#27ae60'] + [GOLD] + ['#e67e22'] + [GOLD, GREEN]
dec_colors = dec_colors[:len(decade_data)]

ax1.barh(decade_labels, decade_data['count'], color=dec_colors, zorder=3)
for i, val in enumerate(decade_data['count']):
    ax1.text(val + 2, i, str(val), va='center', fontsize=9, color='#9999b8')
ax1.set_xlabel('Number of films')
ax1.set_title('Films watched per decade')
ax1.set_facecolor('#111118')

ax2.barh(decade_labels, decade_data['avg_rating'], color=dec_colors, zorder=3)
ax2.axvline(x=ratings['Rating'].mean(), color=GOLD, linestyle='--', linewidth=1, alpha=0.6,
            label=f'Overall avg ({ratings["Rating"].mean():.2f})')
for i, val in enumerate(decade_data['avg_rating']):
    ax2.text(val + 0.05, i, f'{val:.2f}', va='center', fontsize=9, color='#9999b8')
ax2.set_xlabel('Average rating')
ax2.set_title('Average rating per decade')
ax2.set_xlim(0, 5.8)
ax2.set_facecolor('#111118')
ax2.legend(facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_decades.png'), dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/03_decades.png")

# CHART 4 - OVERVIEW

fig = plt.figure(figsize=(12, 6))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('diogocc  //  Letterboxd Cinema Universe', fontsize=16, color=GOLD, y=0.98)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

stats = [
    ('768',  'films watched'),
    ('3.32', 'avg rating'),
    ('107',  '5-star films'),
    ('417',  'reviews written'),
]
card_colors = [GOLD, BLUE, GREEN, PURPLE]

for i, ((val, label), color) in enumerate(zip(stats, card_colors)):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor('#1a1a24')
    ax.text(0.5, 0.65, val, transform=ax.transAxes,
            ha='center', va='center', fontsize=26, fontweight='bold', color=color)
    ax.text(0.5, 0.22, label, transform=ax.transAxes,
            ha='center', va='center', fontsize=9, color='#666680')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a3a')

ax_hist = fig.add_subplot(gs[1, :])
ax_hist.set_facecolor('#111118')
year_counts = ratings['Year'].value_counts().sort_index()
bar_c = [GREEN if y >= 2020 else GOLD if y >= 2010 else '#e67e22' if y >= 2000 else BLUE
         for y in year_counts.index]
ax_hist.bar(year_counts.index, year_counts.values, color=bar_c, width=0.8, zorder=3)
ax_hist.set_xlabel('Release year')
ax_hist.set_ylabel('Films')
ax_hist.set_title('Films by release year')
ax_hist.legend(handles=[
    Patch(color=BLUE,      label='Before 2000'),
    Patch(color='#e67e22', label='2000s'),
    Patch(color=GOLD,      label='2010s'),
    Patch(color=GREEN,     label='2020s'),
], facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8', fontsize=8)

plt.savefig(os.path.join(OUT_DIR, '04_overview.png'), dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/04_overview.png")

# DONE
most_active = diary.groupby('yearmonth').size()
print()
print("=" * 50)
print(" 4 charts saved in output/")
print("=" * 50)
print(f"  Films watched:     {len(watched)}")
print(f"  Avg rating:        {ratings['Rating'].mean():.2f} / 5.0")
print(f"  5-star films:      {(ratings['Rating'] == 5.0).sum()}")
print(f"  0.5-star films:    {(ratings['Rating'] <= 0.5).sum()}")
print(f"  Reviews written:   {len(reviews_with_text)}")
print(f"  Most active month: {most_active.idxmax()} ({most_active.max()} films)")
print(f"  Oldest film:       {ratings.loc[ratings['Year'].idxmin(), 'Name']} ({int(ratings['Year'].min())})")
print(f"  Newest film:       {ratings.loc[ratings['Year'].idxmax(), 'Name']} ({int(ratings['Year'].max())})")
print()