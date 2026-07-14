"""
SCRIPT 07 - Network Model Comparison
======================================
Builds 4 networks with the same n=509 nodes and ~5131 edges:
  - My real film network
  - Erdos-Renyi (random)
  - Barabasi-Albert (scale-free, preferential attachment)
  - Watts-Strogatz (small-world)

Compares them across key network science metrics and generates
visual comparisons: degree distributions, property radar, visualisations.
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'processed', 'movies_enriched.csv')
OUT_DIR = os.path.join(ROOT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor':  '#0a0a0f',
    'axes.facecolor':    '#111118',
    'axes.edgecolor':    '#333344',
    'axes.labelcolor':   '#9999b8',
    'axes.titlecolor':   '#e8e8f0',
    'axes.titlesize':    12,
    'axes.titlepad':     12,
    'axes.grid':         True,
    'grid.color':        '#1e1e2e',
    'grid.linewidth':    0.6,
    'xtick.color':       '#666680',
    'ytick.color':       '#666680',
    'text.color':        '#e8e8f0',
    'font.family':       'monospace',
})

GOLD   = '#e8c96a'
GREEN  = '#00c030'
RED    = '#e84040'
BLUE   = '#4488ff'
PURPLE = '#9966ff'
TEAL   = '#00ccaa'

# ─────────────────────────────────────────────────────────────
# STEP 1 - BUILD THE REAL FILM NETWORK
# ─────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_FILE)
df = df[df['genres'].notna() & df['directors'].notna()].copy()

G_real_full = nx.Graph()
for _, row in df.iterrows():
    G_real_full.add_node(row['Name'])

films = df.to_dict('records')
for i in range(len(films)):
    for j in range(i+1, len(films)):
        a, b = films[i], films[j]
        weight = 0
        weight += len(set(a['directors'].split('|')) & set(b['directors'].split('|'))) * 3
        weight += len(set(a['genres'].split('|')) & set(b['genres'].split('|')))
        if weight >= 3:
            G_real_full.add_edge(a['Name'], b['Name'], weight=weight)

largest_cc = max(nx.connected_components(G_real_full), key=len)
G_real = G_real_full.subgraph(largest_cc).copy()

# Parameters to match in synthetic models
N = G_real.number_of_nodes()   # 509
M = G_real.number_of_edges()   # 5131
p = (2 * M) / (N * (N - 1))   # 0.0397
k = int(round(2 * M / N))      # avg degree ~ 20

print(f"  Real network: {N} nodes, {M} edges, avg degree {k}")

# ─────────────────────────────────────────────────────────────
# STEP 2 - BUILD SYNTHETIC MODELS
# All have same N nodes and approximately same number of edges
# ─────────────────────────────────────────────────────────────

# Erdos-Renyi: each edge exists with probability p
# Result: random graph, Poisson degree distribution, no hubs
G_er = nx.erdos_renyi_graph(N, p, seed=42)
if not nx.is_connected(G_er):
    largest = max(nx.connected_components(G_er), key=len)
    G_er = G_er.subgraph(largest).copy()
print(f"  Erdos-Renyi:    {G_er.number_of_nodes()} nodes, {G_er.number_of_edges()} edges")

# Barabasi-Albert: grows by preferential attachment
# m = edges added per new node — calibrated to match avg degree
m_ba = max(1, k // 2)
G_ba = nx.barabasi_albert_graph(N, m_ba, seed=42)
print(f"  Barabasi-Albert: {G_ba.number_of_nodes()} nodes, {G_ba.number_of_edges()} edges")

# Watts-Strogatz: ring lattice rewired with probability p_ws
# k_ws = neighbours per node in ring, p_ws = rewiring probability
# p_ws=0.1 gives classic small-world regime
k_ws  = k if k % 2 == 0 else k - 1
G_ws = nx.watts_strogatz_graph(N, k_ws, 0.1, seed=42)
print(f"  Watts-Strogatz:  {G_ws.number_of_nodes()} nodes, {G_ws.number_of_edges()} edges")

# ─────────────────────────────────────────────────────────────
# STEP 3 - COMPUTE METRICS FOR ALL 4 NETWORKS
# ─────────────────────────────────────────────────────────────

def compute_metrics(G, name):
    degrees = [d for n, d in G.degree()]
    clust = nx.average_clustering(G)
    path = nx.average_shortest_path_length(G)
    diam = nx.diameter(G)
    bet = list(nx.betweenness_centrality(G).values())
    pr = list(nx.pagerank(G).values())
    density   = nx.density(G)

    # Small-world sigma compared to equivalent ER
    G_rand = nx.erdos_renyi_graph(G.number_of_nodes(),
                                   nx.density(G), seed=99)
    if not nx.is_connected(G_rand):
        lcc = max(nx.connected_components(G_rand), key=len)
        G_rand = G_rand.subgraph(lcc).copy()
    c_rand = nx.average_clustering(G_rand)
    l_rand = nx.average_shortest_path_length(G_rand)
    sigma = (clust / c_rand) / (path / l_rand) if c_rand > 0 else 0

    print(f"  {name}: clustering={clust:.3f}, path={path:.2f}, "
          f"diameter={diam}, sigma={sigma:.2f}")

    return {
        'name':        name,
        'nodes':       G.number_of_nodes(),
        'edges':       G.number_of_edges(),
        'density':     round(density, 4),
        'avg_degree':  round(np.mean(degrees), 2),
        'max_degree':  max(degrees),
        'clustering':  round(clust, 4),
        'avg_path':    round(path, 3),
        'diameter':    diam,
        'sigma':       round(sigma, 2),
        'degrees':     degrees,
        'betweenness': bet,
        'pagerank':    pr,
    }

metrics = {}
for G, name in [
    (G_real, 'My Film Network'),
    (G_er,   'Erdos-Renyi'),
    (G_ba,   'Barabasi-Albert'),
    (G_ws,   'Watts-Strogatz'),
]:
    metrics[name] = compute_metrics(G, name)

# ─────────────────────────────────────────────────────────────
# STEP 4 - VISUALISE
# ─────────────────────────────────────────────────────────────

COLORS = {
    'My Film Network':  GOLD,
    'Erdos-Renyi':      BLUE,
    'Barabasi-Albert':  RED,
    'Watts-Strogatz':   GREEN,
}
NAMES = list(metrics.keys())

# ── CHART 1: Degree distributions (4 panels) ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('Degree Distributions — Real vs Synthetic Models', color=GOLD, fontsize=14)

for ax, name in zip(axes.flat, NAMES):
    degs = metrics[name]['degrees']
    color = COLORS[name]
    ax.hist(degs, bins=30, color=color, alpha=0.85, zorder=3, edgecolor='#0a0a0f')
    ax.axvline(np.mean(degs), color='white', linestyle='--', linewidth=1.2,
               label=f'mean={np.mean(degs):.1f}')
    ax.set_title(name)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_facecolor('#111118')
    ax.legend(facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8', fontsize=9)

    # Annotation
    ax.text(0.97, 0.95,
            f"max={metrics[name]['max_degree']}\nsigma={metrics[name]['sigma']}",
            transform=ax.transAxes, fontsize=8, color='#9999b8',
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a24', edgecolor='#333344'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '23_degree_distributions.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/23_degree_distributions.png")

# ── CHART 2: Metric comparison bar charts ──
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('Network Metrics — My Film Network vs Theoretical Models',
             color=GOLD, fontsize=14)

comparisons = [
    ('clustering',  'Clustering Coefficient\n(higher = more triangles)'),
    ('avg_path',    'Avg Shortest Path Length\n(lower = more connected)'),
    ('sigma',       'Small-World Sigma\n(>1 = small world)'),
    ('max_degree',  'Max Degree\n(hubs)'),
    ('avg_degree',  'Avg Degree'),
    ('diameter',    'Diameter\n(longest shortest path)'),
]

for ax, (metric, title) in zip(axes.flat, comparisons):
    vals   = [metrics[n][metric] for n in NAMES]
    colors = [COLORS[n] for n in NAMES]
    bars   = ax.bar(NAMES, vals, color=colors, zorder=3, alpha=0.85)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(vals)*0.01,
                f'{val:.3f}' if isinstance(val, float) else str(val),
                ha='center', fontsize=8, color='#e8e8f0', fontweight='bold')

    ax.set_title(title)
    ax.set_facecolor('#111118')
    ax.set_xticks(range(len(NAMES)))
    ax.set_xticklabels([n.replace(' ', '\n') for n in NAMES], fontsize=8)

    # Highlight "My Film Network" bar
    bars[0].set_edgecolor(GOLD)
    bars[0].set_linewidth(2)

    # Reference line for sigma
    if metric == 'sigma':
        ax.axhline(y=1, color='#666680', linestyle='--', linewidth=1, alpha=0.7,
                   label='threshold = 1')
        ax.legend(facecolor='#1a1a24', edgecolor='#333344',
                  labelcolor='#9999b8', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '24_metric_comparison.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/24_metric_comparison.png")

# ── CHART 3: Log-log degree distributions (scale-free check) ──
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#111118')
fig.suptitle('Log-Log Degree Distribution — Scale-Free Check\n'
             'A straight line in log-log = power law = scale-free network',
             color=GOLD, fontsize=12)

for name in NAMES:
    degs = metrics[name]['degrees']
    deg_series = pd.Series(degs).value_counts().sort_index()
    deg_series = deg_series[deg_series > 0]
    ax.scatter(deg_series.index, deg_series.values,
               color=COLORS[name], alpha=0.75, s=25,
               label=name, zorder=3)
    # Power law fit
    log_x = np.log(deg_series.index.values.astype(float) + 1e-9)
    log_y = np.log(deg_series.values.astype(float) + 1e-9)
    if len(log_x) > 2:
        coeffs = np.polyfit(log_x, log_y, 1)
        x_fit = np.linspace(deg_series.index.min(), deg_series.index.max(), 100)
        y_fit = np.exp(coeffs[1]) * (x_fit + 1e-9) ** coeffs[0]
        ax.plot(x_fit, y_fit, color=COLORS[name], linewidth=1.2,
                linestyle='--', alpha=0.5)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Degree (log scale)')
ax.set_ylabel('Count (log scale)')
ax.legend(facecolor='#1a1a24', edgecolor='#333344',
          labelcolor='#9999b8', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '25_loglog_comparison.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/25_loglog_comparison.png")

# ── CHART 4: Network visualisations side by side ──
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('Network Topology — Visual Comparison', color=GOLD, fontsize=14)

for ax, (G, name) in zip(axes.flat, [
    (G_real, 'My Film Network'),
    (G_er,   'Erdos-Renyi'),
    (G_ba,   'Barabasi-Albert'),
    (G_ws,   'Watts-Strogatz'),
]):
    ax.set_facecolor('#0a0a0f')
    color = COLORS[name]

    # Sample 120 nodes for readability
    sample = list(G.nodes())[:120]
    G_sub  = G.subgraph(sample).copy()

    pos = nx.spring_layout(G_sub, k=1.5, seed=42)
    degrees_sub = dict(G_sub.degree())

    nx.draw_networkx_edges(G_sub, pos, ax=ax,
        edge_color='#1a1a3a', width=0.4, alpha=0.6)

    nx.draw_networkx_nodes(G_sub, pos, ax=ax,
        node_size=[degrees_sub[n] * 8 for n in G_sub.nodes()],
        node_color=color, alpha=0.85)

    ax.set_title(
        f'{name}\n'
        f'C={metrics[name]["clustering"]:.3f}  '
        f'L={metrics[name]["avg_path"]:.2f}  '
        f'sigma={metrics[name]["sigma"]:.1f}',
        color=color
    )
    ax.set_axis_off()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '26_topology_comparison.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/26_topology_comparison.png")

# ─────────────────────────────────────────────────────────────
# SAVE METRICS TO JSON (for the dashboard later)
# ─────────────────────────────────────────────────────────────
import json

metrics_clean = {
    name: {k: v for k, v in m.items() if k not in ['degrees', 'betweenness', 'pagerank']}
    for name, m in metrics.items()
}
with open(os.path.join(ROOT_DIR, 'data', 'processed', 'network_comparison.json'), 'w') as f:
    json.dump(metrics_clean, f, indent=2)
print("  Metrics saved to data/processed/network_comparison.json")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  DONE! 4 charts saved.")
print("=" * 60)
print()
print(f"  {'Metric':<22} {'My Network':>14} {'ER':>14} {'BA':>14} {'WS':>14}")
print(f"  {'-'*78}")
for metric in ['clustering', 'avg_path', 'diameter', 'sigma', 'max_degree']:
    vals = [f"{metrics[n][metric]:<14}" for n in NAMES]
    print(f"  {metric:<22} {''.join(vals)}")
print()
print("  Verdict:")
m = metrics['My Film Network']
if m['sigma'] > 3:
    print(f"  -> Strong small-world network (sigma={m['sigma']})")
if m['clustering'] > metrics['Erdos-Renyi']['clustering'] * 3:
    print(f"  -> Much higher clustering than random ({m['clustering']:.3f} vs "
          f"{metrics['Erdos-Renyi']['clustering']:.3f})")
if m['max_degree'] > metrics['Erdos-Renyi']['max_degree']:
    print(f"  -> Has hubs like Barabasi-Albert (max degree={m['max_degree']})")
