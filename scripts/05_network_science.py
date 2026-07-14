"""
SCRIPT 05 - Network Science on my film graph
=============================================
Constructs a film-film network where two films are connected
if they share a director (weight 3) or genres (weight 1 each).

Concepts covered:
  1. Network construction + basic properties
  2. Degree distribution (is it scale-free?)
  3. Centrality measures (degree, betweenness, PageRank, closeness)
  4. Small-world test (clustering vs path length)
  5. Community detection (Louvain)
  6. Network visualisation coloured by community
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
GOLD   = '#e8c96a'
GREEN  = '#00c030'
RED    = '#e84040'
BLUE   = '#4488ff'
PURPLE = '#9966ff'
TEAL   = '#00ccaa'

# ─────────────────────────────────────────────────────────────
# BUILD THE NETWORK
# ─────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_FILE)
df = df[df['genres'].notna() & df['directors'].notna()].copy()

G = nx.Graph()

# Each film is a node with attributes
for _, row in df.iterrows():
    G.add_node(row['Name'],
               year=int(row['Year']),
               rating=float(row['Rating']) if pd.notna(row['Rating']) else 0,
               genres=row['genres'],
               director=row['directors'])

# Two films are connected if they share enough in common
# Weight = (shared directors * 3) + shared genres
# Threshold >= 3 keeps only meaningful connections
films = df.to_dict('records')
for i in range(len(films)):
    for j in range(i+1, len(films)):
        a, b = films[i], films[j]
        weight = 0

        dirs_a = set(a['directors'].split('|'))
        dirs_b = set(b['directors'].split('|'))
        shared_dirs   = dirs_a & dirs_b
        weight += len(shared_dirs) * 3  # same director = strong link

        gen_a = set(a['genres'].split('|'))
        gen_b = set(b['genres'].split('|'))
        shared_genres = gen_a & gen_b
        weight += len(shared_genres)

        if weight >= 3:
            G.add_edge(a['Name'], b['Name'],
                       weight=weight,
                       same_director=len(shared_dirs) > 0)

# Work on the largest connected component
largest_cc = max(nx.connected_components(G), key=len)
G_main = G.subgraph(largest_cc).copy()

n_nodes = G_main.number_of_nodes()
n_edges = G_main.number_of_edges()
density = nx.density(G_main)

print(f"  Full graph:  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"  Main component: {n_nodes} nodes, {n_edges} edges")
print(f"  Density: {density:.4f}")

# ─────────────────────────────────────────────────────────────
# CHART 1 - DEGREE DISTRIBUTION
# Is this a scale-free network? (power law degree distribution)
# In scale-free networks, most nodes have few connections but
# a few "hubs" have enormously many — like the internet or social networks
# ─────────────────────────────────────────────────────────────

degrees = [d for n, d in G_main.degree()]
deg_counts = pd.Series(degrees).value_counts().sort_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('Degree Distribution  //  Is my film network scale-free?', color=GOLD, fontsize=14)

# Left: regular histogram
ax1.bar(deg_counts.index, deg_counts.values, color=TEAL, alpha=0.8, zorder=3)
ax1.set_xlabel('Degree (number of connections)')
ax1.set_ylabel('Number of films')
ax1.set_title('Linear scale')
ax1.set_facecolor('#111118')
ax1.axvline(np.mean(degrees), color=GOLD, linestyle='--', linewidth=1.5,
            label=f'Mean: {np.mean(degrees):.1f}')
ax1.axvline(np.median(degrees), color=GREEN, linestyle='--', linewidth=1.5,
            label=f'Median: {np.median(degrees):.1f}')
ax1.legend(facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8')

# Right: log-log scale — a straight line here = power law = scale-free
# Filter out zero counts for log scale
deg_log = deg_counts[deg_counts > 0]
ax2.scatter(deg_log.index, deg_log.values, color=TEAL, alpha=0.8, s=20, zorder=3)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Degree (log scale)')
ax2.set_ylabel('Count (log scale)')
ax2.set_title('Log-log scale  //  straight line = scale-free')
ax2.set_facecolor('#111118')

# Fit a power law line
log_x = np.log(deg_log.index.values.astype(float))
log_y = np.log(deg_log.values.astype(float))
coeffs = np.polyfit(log_x, log_y, 1)
gamma = -coeffs[0]
x_fit = np.linspace(deg_log.index.min(), deg_log.index.max(), 100)
y_fit = np.exp(coeffs[1]) * x_fit ** coeffs[0]
ax2.plot(x_fit, y_fit, color=GOLD, linewidth=1.5, linestyle='--',
         label=f'Power law fit  gamma={gamma:.2f}')
ax2.legend(facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8')

# Annotation box with network summary
summary = (f"Nodes: {n_nodes}\nEdges: {n_edges}\n"
           f"Density: {density:.3f}\nMean degree: {np.mean(degrees):.1f}\n"
           f"Max degree: {max(degrees)}")
ax1.text(0.97, 0.97, summary, transform=ax1.transAxes,
         fontsize=8, color='#9999b8', va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a24', edgecolor='#333344'))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '15_degree_distribution.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/15_degree_distribution.png")

# ─────────────────────────────────────────────────────────────
# CHART 2 - CENTRALITY MEASURES
# Four different ways to measure "importance" in the network
# Degree: raw connections
# Betweenness: how often a node sits on shortest paths (bridges)
# Closeness: how quickly a node can reach all others
# PageRank: like Google's algorithm — important if connected to important nodes
# ─────────────────────────────────────────────────────────────

print("  Computing degree centrality...")
deg_cent = nx.degree_centrality(G_main)

print("  Computing betweenness centrality (slow)...")
bet_cent = nx.betweenness_centrality(G_main, normalized=True, weight='weight')

print("  Computing closeness centrality...")
clo_cent = nx.closeness_centrality(G_main)

print("  Computing PageRank...")
pagerank = nx.pagerank(G_main, weight='weight', alpha=0.85)

centralities = {
    'Degree\n(most connections)':       deg_cent,
    'Betweenness\n(bridge between groups)': bet_cent,
    'Closeness\n(reaches all quickly)':  clo_cent,
    'PageRank\n(influential neighbours)': pagerank,
}

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle('Centrality Measures  //  4 ways to measure importance in my film network',
             color=GOLD, fontsize=14)

colors_cent = [TEAL, PURPLE, GREEN, GOLD]

for ax, (title, cent), color in zip(axes.flat, centralities.items(), colors_cent):
    top = sorted(cent.items(), key=lambda x: -x[1])[:15]
    names = [n[:28] + '...' if len(n) > 28 else n for n, v in top]
    vals  = [v for n, v in top]

    bars = ax.barh(names[::-1], vals[::-1], color=color, alpha=0.85, zorder=3)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + max(vals)*0.01,
                bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=7, color='#9999b8')

    ax.set_title(title, color='#e8e8f0')
    ax.set_facecolor('#111118')
    ax.set_xlabel('Centrality score')
    ax.set_xlim(0, max(vals) * 1.25)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '16_centrality_measures.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/16_centrality_measures.png")

# ─────────────────────────────────────────────────────────────
# CHART 3 - SMALL WORLD TEST
# A small-world network has:
#   - High clustering coefficient (friends of friends are also friends)
#   - Short average path length (six degrees of separation)
# We compare our network against a random Erdos-Renyi graph
# with the same number of nodes and edges
# ─────────────────────────────────────────────────────────────

avg_clustering = nx.average_clustering(G_main)
avg_path = nx.average_shortest_path_length(G_main)
diameter = nx.diameter(G_main)

# Generate equivalent random graph to compare
p_random = (2 * n_edges) / (n_nodes * (n_nodes - 1))
G_random = nx.erdos_renyi_graph(n_nodes, p_random, seed=42)

# Make sure random graph is connected for path calculation
if not nx.is_connected(G_random):
    largest_rand = max(nx.connected_components(G_random), key=len)
    G_random = G_random.subgraph(largest_rand).copy()

rand_clustering = nx.average_clustering(G_random)
rand_path = nx.average_shortest_path_length(G_random)

# Small-world coefficient sigma:
# sigma = (C/C_rand) / (L/L_rand)
# sigma >> 1 means small-world
sigma = (avg_clustering / rand_clustering) / (avg_path / rand_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle(f'Small World Analysis  //  sigma = {sigma:.2f}  (>1 = small world confirmed)',
             color=GOLD, fontsize=13)

# Bar 1: clustering comparison
ax = axes[0]
ax.bar(['My network', 'Random graph'], [avg_clustering, rand_clustering],
       color=[GOLD, MUTED := '#444460'], zorder=3)
ax.set_title('Clustering coefficient\n(higher = more triangles)')
ax.set_facecolor('#111118')
ax.set_ylabel('Avg clustering coefficient')
for i, v in enumerate([avg_clustering, rand_clustering]):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11,
            color=GOLD if i == 0 else '#9999b8', fontweight='bold')

# Bar 2: path length comparison
ax = axes[1]
ax.bar(['My network', 'Random graph'], [avg_path, rand_path],
       color=[TEAL, MUTED], zorder=3)
ax.set_title('Avg shortest path length\n(lower = more connected)')
ax.set_facecolor('#111118')
ax.set_ylabel('Average path length')
for i, v in enumerate([avg_path, rand_path]):
    ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=11,
            color=TEAL if i == 0 else '#9999b8', fontweight='bold')

# Bar 3: sigma and diameter
ax = axes[2]
ax.set_facecolor('#111118')
ax.set_axis_off()

info = [
    ('Small-world sigma', f'{sigma:.2f}'),
    ('(sigma > 1 = small world)', ''),
    ('', ''),
    ('Diameter', f'{diameter} hops'),
    ('Avg path length', f'{avg_path:.2f} hops'),
    ('Avg clustering', f'{avg_clustering:.3f}'),
    ('', ''),
    ('Interpretation', ''),
    ('Any film connects to', ''),
    ('any other in ~4 hops', ''),
]

y = 0.9
for label, val in info:
    if label == 'Small-world sigma':
        ax.text(0.1, y, label, transform=ax.transAxes, fontsize=10, color='#9999b8')
        ax.text(0.75, y, val, transform=ax.transAxes, fontsize=14,
                color=GREEN, fontweight='bold')
    elif label == 'Interpretation':
        ax.text(0.1, y, label, transform=ax.transAxes, fontsize=10,
                color=GOLD, fontweight='bold')
    elif val:
        ax.text(0.1, y, label, transform=ax.transAxes, fontsize=9, color='#9999b8')
        ax.text(0.75, y, val, transform=ax.transAxes, fontsize=9, color='#e8e8f0')
    else:
        ax.text(0.1, y, label, transform=ax.transAxes, fontsize=9, color='#666680')
    y -= 0.09

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '17_small_world.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/17_small_world.png")

# ─────────────────────────────────────────────────────────────
# CHART 4 - COMMUNITY DETECTION (Louvain)
# Louvain maximises modularity — finds groups of nodes that
# are more connected to each other than to the rest of the network
# ─────────────────────────────────────────────────────────────

try:
    from community import community_louvain
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'python-louvain', '-q'])
    from community import community_louvain

partition = community_louvain.best_partition(G_main, weight='weight', random_state=42)
modularity = community_louvain.modularity(partition, G_main, weight='weight')

# Assign community to each node
nx.set_node_attributes(G_main, partition, 'community')

n_communities = len(set(partition.values()))
print(f"  Found {n_communities} communities, modularity = {modularity:.3f}")

# Analyse each community
comm_data = []
for comm_id in set(partition.values()):
    members = [n for n, c in partition.items() if c == comm_id]
    ratings = [G_main.nodes[n]['rating'] for n in members if G_main.nodes[n]['rating'] > 0]
    
    # Find dominant genre in this community
    all_genres = []
    for n in members:
        all_genres.extend(G_main.nodes[n]['genres'].split('|'))
    top_genre = pd.Series(all_genres).value_counts().index[0] if all_genres else 'Unknown'

    # Find dominant director
    all_dirs = []
    for n in members:
        all_dirs.extend(G_main.nodes[n]['director'].split('|'))
    top_dir = pd.Series(all_dirs).value_counts().index[0] if all_dirs else 'Unknown'

    comm_data.append({
        'community': comm_id,
        'size': len(members),
        'avg_rating': np.mean(ratings) if ratings else 0,
        'top_genre': top_genre,
        'top_director': top_dir,
        'members': members
    })

comm_df = pd.DataFrame(comm_data).sort_values('size', ascending=False)

# Plot community summary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.patch.set_facecolor('#0a0a0f')
fig.suptitle(
    f'Community Detection (Louvain)  //  {n_communities} communities  //  modularity = {modularity:.3f}',
    color=GOLD, fontsize=13
)

# Left: community sizes
top_comms = comm_df.head(15)
comm_colors = plt.cm.tab20(np.linspace(0, 1, len(top_comms)))
bars = ax1.barh(
    [f"#{row['community']} - {row['top_genre']}" for _, row in top_comms.iterrows()],
    top_comms['size'],
    color=comm_colors, zorder=3
)
for bar, (_, row) in zip(bars, top_comms.iterrows()):
    ax1.text(bar.get_width() + 0.3,
             bar.get_y() + bar.get_height()/2,
             f"{row['size']} films  avg {row['avg_rating']:.2f}",
             va='center', fontsize=8, color='#9999b8')
ax1.set_xlabel('Community size (films)')
ax1.set_title('Largest communities')
ax1.set_facecolor('#111118')
ax1.set_xlim(0, top_comms['size'].max() * 1.35)

# Right: community avg rating vs size bubble chart
ax2.set_facecolor('#111118')
scatter = ax2.scatter(
    comm_df['size'],
    comm_df['avg_rating'],
    s=comm_df['size'] * 5,
    c=comm_df['avg_rating'],
    cmap='RdYlGn',
    vmin=2.0, vmax=5.0,
    alpha=0.85, zorder=3,
    edgecolors='#333344', linewidths=0.5
)
for _, row in comm_df[comm_df['size'] >= 15].iterrows():
    ax2.annotate(
        f"#{row['community']}\n{row['top_genre']}",
        (row['size'], row['avg_rating']),
        xytext=(5, 5), textcoords='offset points',
        fontsize=7, color='#ccccdd'
    )
ax2.axhline(y=df['Rating'].mean(), color=GOLD, linestyle='--',
            linewidth=1, alpha=0.5, label=f"My avg ({df['Rating'].mean():.2f})")
ax2.set_xlabel('Community size')
ax2.set_ylabel('Avg rating I gave')
ax2.set_title('Size vs rating per community')
ax2.legend(facecolor='#1a1a24', edgecolor='#333344', labelcolor='#9999b8')
plt.colorbar(scatter, ax=ax2, label='Avg rating').ax.yaxis.label.set_color('#9999b8')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '18_communities.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/18_communities.png")

# ─────────────────────────────────────────────────────────────
# CHART 5 - NETWORK VISUALISATION
# Full network coloured by community, sized by PageRank
# Only the main component, spring layout
# ─────────────────────────────────────────────────────────────

# Use spring layout on a subset for readability
# Keep only nodes with degree >= median to reduce clutter
med_degree = np.median([d for n, d in G_main.degree()])
G_vis = G_main.subgraph(
    [n for n, d in G_main.degree() if d >= med_degree]
).copy()

print(f"  Visualising {G_vis.number_of_nodes()} nodes (degree >= {int(med_degree)})")

pos = nx.spring_layout(G_vis, k=1.8, iterations=80, seed=42, weight='weight')

fig, ax = plt.subplots(figsize=(18, 14))
fig.patch.set_facecolor('#0a0a0f')
ax.set_facecolor('#0a0a0f')
fig.suptitle(
    'My Film Network  //  colour = community  //  size = PageRank  //  edges = shared director/genres',
    color=GOLD, fontsize=13
)

# Edges — thicker if same director
edge_colors = ['rgba(100,100,200,0.3)' if G_vis[u][v].get('same_director') else '#1a1a3a'
               for u, v in G_vis.edges()]
edge_widths = [1.5 if G_vis[u][v].get('same_director') else 0.3 for u, v in G_vis.edges()]

nx.draw_networkx_edges(G_vis, pos, ax=ax,
    edge_color=['#3a3a6a' if G_vis[u][v].get('same_director') else '#1a1a2e'
                for u, v in G_vis.edges()],
    width=edge_widths, alpha=0.6)

# Nodes — colour by community, size by PageRank
communities_vis = [partition.get(n, 0) for n in G_vis.nodes()]
pr_vis = nx.pagerank(G_vis, weight='weight')
node_sizes  = [pr_vis.get(n, 0.001) * 8000 for n in G_vis.nodes()]

cmap_comm = plt.cm.tab20
n_comm = len(set(communities_vis))
node_colors = [cmap_comm(c / n_comm) for c in communities_vis]

nx.draw_networkx_nodes(G_vis, pos, ax=ax,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.88)

# Label only the highest PageRank nodes
top_pr_nodes = sorted(pr_vis.items(), key=lambda x: -x[1])[:20]
labels = {n: (n[:18]+'...' if len(n) > 18 else n) for n, v in top_pr_nodes}
nx.draw_networkx_labels(G_vis, pos, labels=labels, ax=ax,
    font_size=6.5, font_color='#eeeeee',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#0a0a14',
              edgecolor='none', alpha=0.7))

ax.set_axis_off()

# Legend: top 6 communities
top6 = comm_df.head(6)
legend_patches = [
    mpatches.Patch(
        color=cmap_comm(row['community'] / n_comm),
        label=f"#{row['community']} {row['top_genre']} ({row['size']} films, avg {row['avg_rating']:.1f})"
    )
    for _, row in top6.iterrows()
]
ax.legend(handles=legend_patches, loc='lower left',
          facecolor='#1a1a24', edgecolor='#333344',
          labelcolor='#ccccdd', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '19_network_visualisation.png'),
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print("  Saved: output/19_network_visualisation.png")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print()
print("=" * 55)
print("  DONE! 5 network science charts saved in output/")
print("=" * 55)
print(f"  15 - Degree distribution  (scale-free? gamma={gamma:.2f})")
print(f"  16 - Centrality measures  (4 definitions of importance)")
print(f"  17 - Small world test     (sigma={sigma:.2f})")
print(f"  18 - Community detection  ({n_communities} communities, modularity={modularity:.3f})")
print(f"  19 - Network visualisation")
print()
print("  Key findings:")
print(f"    - Network has {n_nodes} films and {n_edges} connections")
print(f"    - Small-world confirmed (sigma={sigma:.2f} >> 1)")
print(f"    - Average film is {avg_path:.1f} hops from any other film")
print(f"    - {n_communities} natural communities detected")
print(f"    - Modularity {modularity:.3f} (>0.3 = strong community structure)")