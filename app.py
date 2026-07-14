"""
Cinemaverse — Letterboxd Data Science Dashboard
"""

import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Cinemaverse — diogocc",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# MINIMAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

.stApp { background: #0a0a0f !important; }
[data-testid="stAppViewContainer"] { background: #0a0a0f !important; }
.main .block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 36px;
    font-weight: 900;
    color: #e8e8f0;
    line-height: 1.1;
    margin-bottom: 4px;
}
.section-title em { font-style: italic; color: #e8c96a; }
.section-sub {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #444460;
    letter-spacing: 0.1em;
    margin-bottom: 28px;
}
.film-card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.film-card-title { font-size: 14px; font-weight: 500; color: #e8e8f0; }
.film-card-meta { font-family: 'DM Mono', monospace; font-size: 10px; color: #444460; margin-top: 3px; }
.film-card-rating { font-family: 'DM Mono', monospace; font-size: 12px; color: #e8c96a; margin-top: 4px; }

[data-testid="stMetric"] {
    background: #111118 !important;
    border: 1px solid #1e1e2e !important;
    border-radius: 10px !important;
    padding: 16px !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e8c96a !important;
    border-bottom-color: #e8c96a !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    with open('data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

data = load_data()

films_df   = pd.DataFrame(data['films'])
genres_df  = pd.DataFrame(data['genres'])
dirs_df    = pd.DataFrame(data['directors'])
actors_df  = pd.DataFrame(data['actors'])
monthly_df = pd.DataFrame(data['monthly_activity'])
decades_df = pd.DataFrame(data['decades'])
galaxy_df  = pd.DataFrame(data.get('galaxy', []))
reviews_df = pd.DataFrame(data.get('reviews', []))

GOLD   = '#e8c96a'
GREEN  = '#00c030'
RED    = '#e84040'
TEAL   = '#00ccaa'
BG     = '#0a0a0f'
BG2    = '#111118'

def plotly_layout(**kwargs):
    base = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        font=dict(color='#9999b8', family='DM Mono, monospace', size=11),
        xaxis=dict(gridcolor='#1a1a2e', zerolinecolor='#1a1a2e', color='#444460'),
        yaxis=dict(gridcolor='#1a1a2e', zerolinecolor='#1a1a2e', color='#444460'),
        hoverlabel=dict(bgcolor='#13131f', bordercolor=GOLD,
                        font=dict(color='#e8e8f0', family='DM Mono, monospace')),
        margin=dict(l=10, r=10, t=40, b=10),
        title_font=dict(color='#9999b8', family='DM Mono, monospace', size=11),
    )
    base.update(kwargs)
    return base

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
SECTIONS = [
    ("Overview",        "◈"),
    ("Exploration",     "◉"),
    ("Networks",        "⬡"),
    ("Galaxy",          "✦"),
    ("Recommendations", "→"),
    ("Predictions",     "◎"),
    ("Reviews",         "✎"),
    ("Lists",           "▤"),
    ("Directors",       "✦"),
]

if 'section' not in st.session_state:
    st.session_state['section'] = 'Overview'

with st.sidebar:
    st.title("🎬 cinemaverse")
    st.caption("diogocc · letterboxd")
    st.divider()

    for name, icon in SECTIONS:
        if st.button(
            f"{icon}  {name}",
            key=f"nav_{name}",
            use_container_width=True,
            type="primary" if st.session_state['section'] == name else "secondary",
        ):
            st.session_state['section'] = name
            st.rerun()

    st.divider()
    st.caption(f"{data['stats']['total_watched']} films · {data['stats']['first_log'][:4]}–{data['stats']['last_log'][:4]}")

section = st.session_state['section']

# ─────────────────────────────────────────────────────────────
# SECTION 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────
if section == "Overview":
    s = data['stats']
    st.markdown(f"""
    <div class="section-title">{s['total_watched']} films.<br><em>A life</em> watching.</div>
    <div class="section-sub">◈ OVERVIEW // EVERY FILM I'VE WATCHED — IN NUMBERS</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Films",      s['total_watched'])
    c2.metric("Avg rating", s['avg_rating'])
    c3.metric("5-star",     s['five_star_count'])
    c4.metric("0.5-star",   s['half_star_count'])
    c5.metric("Reviews",    s['total_reviews'])
    c6.metric("Watchlist",  s['watchlist_count'])
    c7.metric("Active since", s['first_log'][:4])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        dist   = data['rating_distribution']
        keys   = list(dist.keys())
        vals   = list(dist.values())
        colors = [RED if float(k) <= 1.5 else '#2a2a4a' if float(k) <= 3.0
                  else GOLD if float(k) <= 4.0 else GREEN for k in keys]
        fig = go.Figure(go.Bar(
            x=keys, y=vals, marker_color=colors,
            text=vals, textposition='outside',
            textfont=dict(color='#444460', size=9),
            hovertemplate='%{x}★: %{y} films<extra></extra>'
        ))
        fig.update_layout(**plotly_layout(title='Rating Distribution  //  avg 3.32', height=300, showlegend=False))
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        colors_m = [GREEN if c >= 40 else GOLD if c >= 20 else '#1a1a2e' for c in monthly_df['count']]
        fig = go.Figure(go.Bar(
            x=monthly_df['yearmonth'], y=monthly_df['count'],
            marker_color=colors_m,
            hovertemplate='%{x}: %{y} films<extra></extra>'
        ))
        fig.update_layout(**plotly_layout(title='Monthly Activity  //  peak: Apr 2025 (46 films)', height=300, showlegend=False))
        fig.update_xaxes(title=None, tickangle=45, tickfont=dict(size=8))
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = go.Figure(go.Bar(
            x=decades_df['count'],
            y=[f"{int(d)}s" for d in decades_df['decade']],
            orientation='h', marker_color=TEAL,
            hovertemplate='%{y}: %{x} films<extra></extra>'
        ))
        fig.update_layout(**plotly_layout(title='Films Watched per Decade', height=320, showlegend=False))
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        colors_d = [GREEN if r >= 4.0 else GOLD if r >= 3.5 else TEAL for r in decades_df['avg_rating']]
        fig = go.Figure(go.Bar(
            x=decades_df['avg_rating'],
            y=[f"{int(d)}s" for d in decades_df['decade']],
            orientation='h', marker_color=colors_d,
            hovertemplate='%{y}: avg %{x:.2f}<extra></extra>'
        ))
        fig.add_vline(x=s['avg_rating'], line_dash='dash', line_color='#333344',
                      annotation_text=f"avg {s['avg_rating']}", annotation_font_color='#444460')
        fig.update_layout(**plotly_layout(title='Avg Rating per Decade', height=320, showlegend=False))
        fig.update_xaxes(range=[0, 5.5], title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, use_container_width=True)

    crowd = data['me_vs_crowd']
    st.subheader("Me vs The Crowd")
    ca, cb, cc = st.columns(3)
    ca.metric("Avg bias vs TMDb", f"{crowd['avg_bias']:+.2f}★")
    cb.metric("I rate higher on", f"{crowd['higher_count']} films")
    cc.metric("I rate lower on",  f"{crowd['lower_count']} films")

    col5, col6 = st.columns(2)
    with col5:
        loved_df = pd.DataFrame(crowd['loved_more'])
        if not loved_df.empty:
            fig = go.Figure(go.Bar(
                x=loved_df['diff'],
                y=[f"{r['Name']} ({int(r['Year'])})" for _, r in loved_df.iterrows()],
                orientation='h', marker_color=GREEN,
                hovertemplate='%{y}<br>+%{x:.1f} vs crowd<extra></extra>'
            ))
            fig.update_layout(**plotly_layout(title='Films I loved WAY more than the crowd', height=300, showlegend=False))
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        hated_df = pd.DataFrame(crowd['hated_more'])
        if not hated_df.empty:
            fig = go.Figure(go.Bar(
                x=hated_df['diff'].abs(),
                y=[f"{r['Name']} ({int(r['Year'])})" for _, r in hated_df.iterrows()],
                orientation='h', marker_color=RED,
                hovertemplate='%{y}<br>-%{x:.1f} vs crowd<extra></extra>'
            ))
            fig.update_layout(**plotly_layout(title='Films I hated WAY more than the crowd', height=300, showlegend=False))
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SECTION 2 — EXPLORATION
# ─────────────────────────────────────────────────────────────
elif section == "Exploration":
    st.markdown("""
    <div class="section-title">Genre &amp; Director<br><em>Deep Dive</em></div>
    <div class="section-sub">◉ EXPLORATION // GENRES · DIRECTORS · ACTORS · COMPANIES</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["GENRES", "DIRECTORS", "ACTORS", "COMPANIES"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                genres_df.sort_values('count').tail(20),
                x='count', y='genre', orientation='h',
                color='avg_rating', color_continuous_scale='RdYlGn',
                range_color=[2.5, 4.5], hover_data=['avg_rating'],
                title='Films watched per genre (top 20)'
            )
            fig.update_layout(**plotly_layout(height=500))
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(
                genres_df[genres_df['count'] >= 10].sort_values('avg_rating'),
                x='avg_rating', y='genre', orientation='h',
                color='avg_rating', color_continuous_scale='RdYlGn',
                range_color=[2.5, 4.5],
                title='Avg rating per genre (min 10 films)'
            )
            fig.add_vline(x=data['stats']['avg_rating'], line_dash='dash',
                          line_color='#333344', annotation_text="my avg",
                          annotation_font_color='#444460')
            fig.update_layout(**plotly_layout(height=500))
            fig.update_xaxes(range=[0, 5.5], title=None)
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.scatter(
            dirs_df[dirs_df['count'] >= 3], x='count', y='avg_rating',
            size='count', color='avg_rating', hover_name='director',
            color_continuous_scale='RdYlGn', range_color=[2.5, 5.0], size_max=40,
            title='Directors  //  x = films  //  y = avg rating  //  hover for name'
        )
        fig.add_hline(y=data['stats']['avg_rating'], line_dash='dash',
                      line_color='#333344', annotation_text="my avg",
                      annotation_font_color='#444460')
        fig.update_layout(**plotly_layout(height=480))
        fig.update_xaxes(title='Films watched')
        fig.update_yaxes(title='Avg rating')
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.caption("TOP RATED DIRECTORS (min 3 films)")
            st.dataframe(
                dirs_df[dirs_df['count'] >= 3].nlargest(15, 'avg_rating')
                [['director','count','avg_rating']].rename(columns={
                    'director':'Director','count':'Films','avg_rating':'Avg ★'
                }),
                use_container_width=True, hide_index=True
            )
        with col2:
            st.caption("MOST WATCHED DIRECTORS")
            st.dataframe(
                dirs_df.nlargest(15, 'count')
                [['director','count','avg_rating']].rename(columns={
                    'director':'Director','count':'Films','avg_rating':'Avg ★'
                }),
                use_container_width=True, hide_index=True
            )

    with tab3:
        top_actors = actors_df[actors_df['count'] >= 5].sort_values('count', ascending=False).head(25)
        colors_a = [GREEN if r >= 4.0 else GOLD if r >= 3.5 else TEAL if r >= 3.0 else RED
                    for r in top_actors['avg_rating']]
        fig = go.Figure(go.Bar(
            x=top_actors['count'], y=top_actors['actor'],
            orientation='h', marker_color=colors_a,
            text=[f"avg {r:.1f}" for r in top_actors['avg_rating']],
            textposition='outside', textfont=dict(color='#444460', size=9),
            hovertemplate='%{y}<br>%{x} films  avg %{text}<extra></extra>'
        ))
        fig.update_layout(**plotly_layout(title='Most frequent actors  //  colour = avg rating', height=600, showlegend=False))
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if 'companies' in data:
            comp_df = pd.DataFrame(data['companies'])
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    comp_df.sort_values('count').tail(20),
                    x='count', y='company', orientation='h',
                    color='avg_rating', color_continuous_scale='RdYlGn',
                    range_color=[2.5, 4.5],
                    title='Most watched production companies'
                )
                fig.update_layout(**plotly_layout(height=500))
                fig.update_xaxes(title=None)
                fig.update_yaxes(title=None)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(
                    comp_df[comp_df['count'] >= 3].sort_values('avg_rating').tail(20),
                    x='avg_rating', y='company', orientation='h',
                    color='avg_rating', color_continuous_scale='RdYlGn',
                    range_color=[2.5, 4.5],
                    title='Best rated production companies (min 3 films)'
                )
                fig.add_vline(x=data['stats']['avg_rating'], line_dash='dash', line_color='#333344')
                fig.update_layout(**plotly_layout(height=500))
                fig.update_xaxes(range=[0, 5.5], title=None)
                fig.update_yaxes(title=None)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run script 11a to fetch production companies data.")

# ─────────────────────────────────────────────────────────────
# SECTION 3 — NETWORKS
# ─────────────────────────────────────────────────────────────
elif section == "Networks":
    st.markdown("""
    <div class="section-title">Network<br><em>Science</em></div>
    <div class="section-sub">⬡ NETWORKS // FILM GRAPH · CENTRALITY · SMALL WORLD · COMMUNITIES</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["METRICS COMPARISON", "DIRECTOR NETWORK", "THEORETICAL MODELS"])

    with tab1:
        if data['network']:
            net = data['network']
            networks = list(net.keys())
            metric_labels = {
                'clustering': 'Clustering Coefficient',
                'avg_path':   'Avg Path Length',
                'sigma':      'Small-World Sigma',
                'max_degree': 'Max Degree',
                'density':    'Density',
            }
            selected_metric = st.selectbox("Compare metric", list(metric_labels.keys()),
                                           format_func=lambda x: metric_labels[x])
            vals = {n: net[n].get(selected_metric, 0) for n in networks}
            fig = go.Figure(go.Bar(
                x=list(vals.keys()), y=list(vals.values()),
                marker_color=[GOLD if n == 'My Film Network' else '#1a1a2e' for n in vals],
                marker_line_color=[GOLD if n == 'My Film Network' else '#333344' for n in vals],
                marker_line_width=1.5,
                text=[f"{v:.3f}" if isinstance(v, float) else str(v) for v in vals.values()],
                textposition='outside', textfont=dict(color='#666680', size=10),
                hovertemplate='%{x}: %{y}<extra></extra>'
            ))
            if selected_metric == 'sigma':
                fig.add_hline(y=1, line_dash='dash', line_color=RED,
                              annotation_text='threshold = 1', annotation_font_color=RED)
            fig.update_layout(**plotly_layout(title=metric_labels[selected_metric], height=380, showlegend=False))
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, use_container_width=True)

            my_net = net.get('My Film Network', {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Nodes",    my_net.get('nodes', '—'))
            c2.metric("Edges",    my_net.get('edges', '—'))
            c3.metric("Sigma",    f"{my_net.get('sigma', 0):.2f}")
            c4.metric("Avg path", f"{my_net.get('avg_path', 0):.2f}")

            st.divider()
            sigma = my_net.get('sigma', 0)
            if sigma > 3:
                st.success(f"Strong small-world network (sigma={sigma:.2f})")
            elif sigma > 1:
                st.info(f"Small-world network (sigma={sigma:.2f})")
            else:
                st.warning(f"Not a small-world network (sigma={sigma:.2f})")
        else:
            st.info("Run script 07_network_comparison.py first.")

    with tab2:
        graph = data.get('network_graph', {})
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])

        if nodes:
            st.caption(f"{len(nodes)} directors · {len(edges)} connections · colour = avg rating · size = films watched")

            edge_x, edge_y = [], []
            for e in edges:
                edge_x += [e['x0'], e['x1'], None]
                edge_y += [e['y0'], e['y1'], None]

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, mode='lines',
                line=dict(color='#1a1a3a', width=0.8),
                hoverinfo='none'
            )

            node_trace = go.Scatter(
                x=[n['x'] for n in nodes],
                y=[n['y'] for n in nodes],
                mode='markers+text',
                marker=dict(
                    size=[max(8, min(30, n['films'] * 5)) for n in nodes],
                    color=[n['avg_rating'] for n in nodes],
                    colorscale='RdYlGn',
                    cmin=2.5, cmax=5.0,
                    colorbar=dict(
                        title=dict(text='Avg rating', font=dict(color='#9999b8')),
                        tickfont=dict(color='#9999b8'),
                        bgcolor='#111118',
                    ),
                    line=dict(color='#0a0a0f', width=1),
                ),
                text=[n['id'] for n in nodes],
                textposition='top center',
                textfont=dict(size=7, color='#666680'),
                hovertext=[f"<b>{n['id']}</b><br>{n['films']} films<br>avg {n['avg_rating']:.2f}" for n in nodes],
                hovertemplate='%{hovertext}<extra></extra>',
            )

            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(**plotly_layout(
                title='Director Network  //  connected by shared actors  //  hover for details',
                height=650, showlegend=False,
            ))
            fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
            fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run script 11_export_data.py to generate network graph data.")

    with tab3:
        import networkx as nx

        st.caption("Synthetic networks generated with the same parameters as my real network (509 nodes)")

        model = st.selectbox("Select model", [
            "Erdős–Rényi (random)",
            "Barabási–Albert (scale-free)",
            "Watts–Strogatz (small-world)",
        ])

        # Parameters matching real network
        N = 509
        M = 5131
        p = (2 * M) / (N * (N - 1))
        k = max(1, int(round(2 * M / N)))

        if model == "Erdős–Rényi (random)":
            G = nx.erdos_renyi_graph(N, p, seed=42)
            desc = f"Each pair of nodes connected with probability p={p:.4f}. No hubs, Poisson degree distribution."
        elif model == "Barabási–Albert (scale-free)":
            G = nx.barabasi_albert_graph(N, max(1, k // 2), seed=42)
            desc = "New nodes attach preferentially to well-connected nodes. Creates hubs — power-law degree distribution."
        else:
            k_ws = k if k % 2 == 0 else k - 1
            G = nx.watts_strogatz_graph(N, k_ws, 0.1, seed=42)
            desc = "Ring lattice rewired with p=0.1. High clustering + short paths = small-world."

        st.info(desc)

        # Layout
        pos = nx.spring_layout(G, k=1.5, iterations=30, seed=42)
        degrees = dict(G.degree())

        edge_x, edge_y = [], []
        for u, v in list(G.edges())[:500]:  # limit edges for performance
            edge_x += [pos[u][0], pos[v][0], None]
            edge_y += [pos[u][1], pos[v][1], None]

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_deg = [degrees[n] for n in G.nodes()]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode='lines',
            line=dict(color='#1a1a3a', width=0.5),
            hoverinfo='none'
        )
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers',
            marker=dict(
                size=[max(3, min(15, d)) for d in node_deg],
                color=node_deg,
                colorscale='YlOrRd',
                colorbar=dict(
                    title=dict(text='Degree', font=dict(color='#9999b8')),
                    tickfont=dict(color='#9999b8'),
                ),
                line=dict(color='#0a0a0f', width=0.5),
            ),
            hovertemplate='degree: %{marker.color}<extra></extra>',
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(**plotly_layout(
            title=f'{model}  //  {N} nodes  //  size = degree  //  colour = degree',
            height=600, showlegend=False,
        ))
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

        # Quick stats
        lcc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc).copy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Clustering", f"{nx.average_clustering(G):.3f}")
        c2.metric("Avg path (LCC)", f"{nx.average_shortest_path_length(G_lcc):.2f}")
        c3.metric("Max degree", max(node_deg))

elif section == "Galaxy":
    st.markdown("""
    <div class="section-title">Film<br><em>Galaxy</em></div>
    <div class="section-sub">✦ GALAXY // UMAP · HDBSCAN CLUSTERS · 768 FILMS IN 2D SPACE</div>
    """, unsafe_allow_html=True)

    if not galaxy_df.empty:
        col_ctrl1, col_ctrl2, _ = st.columns([1, 1, 3])
        with col_ctrl1:
            colour_by = st.selectbox("Colour by", ["My Rating", "Cluster", "Decade"])
        with col_ctrl2:
            show_labels = st.toggle("5-star labels", value=True)

        galaxy_merged = galaxy_df.merge(
            films_df[['Name', 'Year', 'genres', 'directors']],
            on=['Name', 'Year'], how='left'
        )

        def make_hover(r):
            genres = str(r.get('genres_y', r.get('genres_x', ''))).replace('|', ' · ')[:40]
            return f"<b>{r['Name']}</b> ({int(r['Year'])})<br>★ {r['Rating']}<br>{genres}"

        galaxy_merged['hover'] = galaxy_merged.apply(make_hover, axis=1)

        if colour_by == "My Rating":
            fig = px.scatter(
                galaxy_merged, x='umap_x', y='umap_y',
                color='Rating', color_continuous_scale='RdYlGn',
                range_color=[0.5, 5.0], size='Rating', size_max=10,
                custom_data=['hover'],
            )
        elif colour_by == "Cluster":
            galaxy_merged['Cluster'] = galaxy_merged['cluster'].astype(str)
            fig = px.scatter(
                galaxy_merged, x='umap_x', y='umap_y',
                color='Cluster', size='Rating', size_max=10,
                custom_data=['hover'],
            )
        else:
            galaxy_merged['Decade'] = (galaxy_merged['Year'] // 10 * 10).astype(str) + 's'
            fig = px.scatter(
                galaxy_merged, x='umap_x', y='umap_y',
                color='Decade', size='Rating', size_max=10,
                custom_data=['hover'],
            )

        fig.update_traces(hovertemplate='%{customdata[0]}<extra></extra>')

        if show_labels:
            for _, row in galaxy_merged[galaxy_merged['Rating'] == 5.0].iterrows():
                fig.add_annotation(
                    x=row['umap_x'], y=row['umap_y'],
                    text=row['Name'][:18],
                    showarrow=False,
                    font=dict(color=GOLD, size=7, family='DM Mono'),
                    yshift=8
                )

        fig.update_layout(**plotly_layout(
            title=f'Film Galaxy — {colour_by}  //  hover any point for details',
            height=620, xaxis_title='UMAP dimension 1', yaxis_title='UMAP dimension 2',
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Point size = my rating. Films close together share similar genres and director.")
    else:
        st.info("Run script 06_embeddings_clustering.py first.")

# ─────────────────────────────────────────────────────────────
# SECTION 5 — RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
elif section == "Recommendations":
    st.markdown("""
    <div class="section-title">What to Watch<br><em>Next</em></div>
    <div class="section-sub">→ RECOMMENDATIONS // COSINE SIMILARITY · SEMANTIC EMBEDDINGS</div>
    """, unsafe_allow_html=True)

    recs = data['recommendations']
    tab1, tab2 = st.tabs(["WATCHLIST", "DISCOVERIES"])

    for tab, cosine_key, semantic_key in [
        (tab1, 'watchlist_cosine',   'watchlist_semantic'),
        (tab2, 'discoveries_cosine', 'discoveries_semantic'),
    ]:
        with tab:
            col_m, col_n, _ = st.columns([1, 1, 3])
            with col_m:
                method = st.radio("Method", ["Cosine", "Semantic"],
                                  horizontal=True, key=f"m_{cosine_key}")
            with col_n:
                top_n = st.slider("Top N", 5, 30, 15, key=f"n_{cosine_key}")

            key = cosine_key if method == "Cosine" else semantic_key
            if key not in recs:
                key = cosine_key

            recs_df   = pd.DataFrame(recs[key])
            score_col = 'similarity_pct' if 'similarity_pct' in recs_df.columns else 'semantic_score'
            top       = recs_df.nlargest(top_n, score_col).reset_index(drop=True)

            s_min, s_max = top[score_col].min(), top[score_col].max()
            top['score_norm'] = ((top[score_col] - s_min) / (s_max - s_min + 1e-9) * 100)

            fig = go.Figure(go.Bar(
                x=top['score_norm'],
                y=[f"{r['Name']}  ({int(r['Year'])})" for _, r in top.iterrows()],
                orientation='h',
                marker_color=[GREEN if s >= 75 else GOLD if s >= 50 else TEAL for s in top['score_norm']],
                hovertemplate='%{y}<br>Match: %{x:.0f}%<extra></extra>',
                text=[str(r.get('genres', '')).split('|')[0][:20] for _, r in top.iterrows()],
                textposition='outside', textfont=dict(color='#333350', size=8),
            ))
            fig.update_layout(**plotly_layout(
                title=f"Top {top_n} — {method} similarity",
                height=max(400, top_n * 30), showlegend=False
            ))
            fig.update_xaxes(range=[0, 130], title=None, showgrid=False)
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# SECTION 6 — PREDICTIONS
# ─────────────────────────────────────────────────────────────
elif section == "Predictions":
    st.markdown("""
    <div class="section-title">Rating<br><em>Predictions</em></div>
    <div class="section-sub">◎ PREDICTIONS // RANDOM FOREST · SEARCH MY FILMS</div>
    """, unsafe_allow_html=True)

    search = st.text_input("", placeholder="Search — e.g. Dune, Parasite, The Batman...",
                           label_visibility="collapsed")

    if search:
        mask    = films_df['Name'].str.contains(search, case=False, na=False)
        results = films_df[mask].head(8)
        if not results.empty:
            cols = st.columns(min(len(results), 4))
            for col, (_, row) in zip(cols * 2, results.iterrows()):
                rating = row.get('Rating')
                genres = str(row.get('genres', '')).replace('|', ' · ')[:35]
                dirs   = str(row.get('directors', '')).split('|')[0][:25]
                stars  = ''
                if pd.notna(rating):
                    stars = '★' * int(rating) + ('½' if rating % 1 >= 0.5 else '')
                col.markdown(f"""
                <div class="film-card">
                    <div class="film-card-title">{row['Name']}</div>
                    <div class="film-card-meta">{int(row['Year'])} · {dirs}</div>
                    <div class="film-card-meta">{genres}</div>
                    <div class="film-card-rating">{stars if stars else 'not rated'}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"No films found for '{search}'")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.caption("MY 5-STAR FILMS")
        five_df = pd.DataFrame(data['five_star'])
        st.dataframe(
            five_df[['Name','Year','genres','directors']].rename(columns={
                'Name':'Film','Year':'Year','genres':'Genres','directors':'Director'
            }),
            use_container_width=True, hide_index=True, height=420
        )
    with col2:
        st.caption("MY 0.5-STAR FILMS")
        half_df = pd.DataFrame(data['half_star'])
        st.dataframe(
            half_df[['Name','Year','genres','directors']].rename(columns={
                'Name':'Film','Year':'Year','genres':'Genres','directors':'Director'
            }),
            use_container_width=True, hide_index=True, height=420
        )

# ─────────────────────────────────────────────────────────────
# SECTION 7 — REVIEWS
# ─────────────────────────────────────────────────────────────
elif section == "Reviews":
    st.markdown(f"""
    <div class="section-title">My<br><em>Reviews</em></div>
    <div class="section-sub">✎ REVIEWS // {data['stats']['total_reviews']} REVIEWS WRITTEN IN PT & EN</div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["ALL REVIEWS", "STATS", "GENERATE WITH AI"])

    # ── TAB 1: All reviews with filters ──
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            min_rating = st.slider("Min rating", 0.5, 5.0, 3.5, 0.5)
        with col2:
            min_length = st.slider("Min length (chars)", 0, 500, 80, 40)
        with col3:
            search_r = st.text_input("", placeholder="Search in text...",
                                     label_visibility="collapsed")

        filtered = reviews_df.copy()
        if 'Rating' in filtered.columns:
            filtered = filtered[filtered['Rating'] >= min_rating]
        filtered = filtered[filtered['review_length'] >= min_length]
        if search_r:
            filtered = filtered[
                filtered['Review'].str.contains(search_r, case=False, na=False)
            ]

        st.caption(f"{len(filtered)} reviews match · showing first 15")

        for _, row in filtered.head(15).iterrows():
            rating  = row.get('Rating')
            stars   = '★' * int(rating) + ('½' if rating % 1 >= 0.5 else '') if pd.notna(rating) else ''
            preview = str(row['Review'])[:400] + ('...' if len(str(row['Review'])) > 400 else '')
            st.markdown(f"""
            <div class="film-card">
                <div style="display:flex; justify-content:space-between; align-items:baseline;">
                    <div class="film-card-title">{row['Name']} ({int(row['Year'])})</div>
                    <div class="film-card-rating">{stars}</div>
                </div>
                <div style="font-size:13px; color:#666680; margin-top:10px;
                            line-height:1.65; font-style:italic;">"{preview}"</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: Review stats ──
    with tab2:
        if not reviews_df.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total reviews", len(reviews_df))
            col2.metric("Avg length", f"{reviews_df['review_length'].mean():.0f} chars")
            col3.metric("Longest review", f"{reviews_df['review_length'].max()} chars")

            st.divider()

            # Review length distribution
            fig = go.Figure(go.Histogram(
                x=reviews_df['review_length'],
                nbinsx=30,
                marker_color=TEAL,
                hovertemplate='Length: %{x}<br>Count: %{y}<extra></extra>'
            ))
            fig.update_layout(**plotly_layout(
                title='Review length distribution (characters)',
                height=300, showlegend=False
            ))
            fig.update_xaxes(title='Characters')
            fig.update_yaxes(title='Reviews')
            st.plotly_chart(fig, use_container_width=True)

            # Top 5 longest reviews
            st.subheader("My longest reviews")
            top_long = reviews_df.nlargest(5, 'review_length')[['Name', 'Year', 'Rating', 'review_length']]
            for _, row in top_long.iterrows():
                rating = row.get('Rating')
                stars  = '★' * int(rating) + ('½' if rating % 1 >= 0.5 else '') if pd.notna(rating) else ''
                st.markdown(f"""
                <div class="film-card">
                    <div class="film-card-title">{row['Name']} ({int(row['Year'])})
                        <span class="film-card-rating"> {stars}</span>
                    </div>
                    <div class="film-card-meta">{int(row['review_length'])} characters</div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: Generate review with GPT-2 ──
    with tab3:
        st.markdown("""
        <div style="font-size:13px; color:#666680; margin-bottom:16px; line-height:1.6;">
        This generator uses <b style="color:#9999b8;">DistilGPT-2</b> fine-tuned on my 417 Letterboxd reviews.
        Before generating, it's important to understand what this model actually learned — and what it didn't.
        </div>
        """, unsafe_allow_html=True)

        st.warning("⚠️  This is a creative experiment with known limitations — read below before generating.")

        with st.expander("Understanding the model's limitations", expanded=True):
            st.markdown("""
**What fine-tuning actually does**

Fine-tuning a language model on your reviews does *not* teach it about films. It teaches it to imitate
your writing patterns — your sentence structures, your vocabulary, your rhythm. Think of it as training
a parrot to speak in your accent, not to understand what it's saying.

---

**Why the output may be factually wrong**

The model has zero factual knowledge about any film. It doesn't know who directed Nosferatu, who acts in it,
or whether it's from 1922 or 2024. When it mentions an actor or a plot detail, it's making a statistically
plausible guess based on words that often appeared together in my reviews — not recalling real information.
This is called **hallucination**, and it's a fundamental limitation of generative language models
when used without grounding in real data.

---

**Why 417 reviews isn't enough**

GPT-2 was pre-trained on billions of words. Fine-tuning it on 417 reviews is like teaching a concert
pianist one new song — the underlying technique changes very little. The model will largely default to
general English patterns with a thin veneer of my style on top. A dataset of tens of thousands of reviews
would be needed to meaningfully shift the model's behaviour.

---

**Why mixing Portuguese and English hurts quality**

My reviews are written in both languages. The model learned from this mixed signal, which means it
sometimes switches languages mid-sentence or produces incoherent text when the two languages conflict
in its internal representations.

---

**What the model actually does well**

Despite these limitations, the experiment is genuinely interesting:
- It captures the **general tone** of my reviews — more analytical for high ratings, more dismissive for low ones
- It tends to use **cinematic vocabulary** similar to mine
- It reflects the **emotional register** I use (the 5★ outputs feel different from the 0.5★ ones)
- It's a real demonstration of **NLP fine-tuning** on a personal dataset

Treat the output as a creative experiment in machine learning — not as a real review.
            """)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            film_input  = st.text_input("Film title", placeholder="e.g. Nosferatu")
        with col2:
            year_input  = st.number_input("Year", min_value=1900, max_value=2030,
                                          value=2024, step=1)
        with col3:
            rating_input = st.select_slider(
                "Rating", options=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], value=4.0
            )

        temp = st.slider("Creativity (temperature)", 0.5, 1.2, 0.8, 0.05,
                         help="Lower = more predictable, Higher = more creative")

        if st.button("Generate review", type="primary"):
            if not film_input:
                st.warning("Please enter a film title.")
            else:
                with st.spinner("Loading model and generating review..."):
                    try:
                        from transformers import GPT2Tokenizer, GPT2LMHeadModel
                        import torch

                        @st.cache_resource
                        def load_gpt2():
                            tok = GPT2Tokenizer.from_pretrained("diogocc45/letterboxd-gpt2")
                            mod = GPT2LMHeadModel.from_pretrained("diogocc45/letterboxd-gpt2")
                            mod.eval()
                            return tok, mod

                        tokenizer, model = load_gpt2()

                        prompt = (f"<|film|> {film_input} ({int(year_input)}) "
                                  f"<|rating|> {rating_input:.1f} <|review|>")

                        inputs = tokenizer.encode(prompt, return_tensors='pt')
                        with torch.no_grad():
                            outputs = model.generate(
                                inputs,
                                max_length=250,
                                temperature=temp,
                                do_sample=True,
                                top_k=50,
                                top_p=0.95,
                                repetition_penalty=1.2,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                            )

                        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
                        if '<|review|>' in generated:
                            review_text = generated.split('<|review|>')[1]
                            review_text = review_text.replace('<|endoftext|>', '').strip()
                        else:
                            review_text = generated

                        stars = '★' * int(rating_input) + ('½' if rating_input % 1 >= 0.5 else '')
                        st.markdown(f"""
                        <div class="film-card">
                            <div style="display:flex; justify-content:space-between;">
                                <div class="film-card-title">{film_input} ({int(year_input)})</div>
                                <div class="film-card-rating">{stars}</div>
                            </div>
                            <div style="font-family:'DM Mono',monospace; font-size:9px;
                                        color:#333350; margin: 4px 0 10px;
                                        letter-spacing:0.1em;">AI-GENERATED IN MY STYLE</div>
                            <div style="font-size:13px; color:#9999b8; line-height:1.65;
                                        font-style:italic;">"{review_text}"</div>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Error loading model: {e}")

# ─────────────────────────────────────────────────────────────
# SECTION 8 — LISTS
# ─────────────────────────────────────────────────────────────
elif section == "Lists":
    st.markdown("""
    <div class="section-title">My<br><em>Lists</em></div>
    <div class="section-sub">▤ LISTS // CURATED COLLECTIONS FROM LETTERBOXD</div>
    """, unsafe_allow_html=True)

    lists_data = data.get('lists', {})

    if not lists_data:
        st.info("Run scripts 11b_fetch_posters.py and 11_export_data.py to generate lists data.")
    else:
        list_names     = list(lists_data.keys())
        selected_list  = st.selectbox("Select a list", list_names)
        films          = lists_data[selected_list]

        st.caption(f"{len(films)} films · click a poster to see details")
        st.divider()

        # Session state for selected film
        if 'selected_film' not in st.session_state:
            st.session_state['selected_film'] = None

        # Poster grid — 6 columns
        COLS = 6
        rows = [films[i:i+COLS] for i in range(0, len(films), COLS)]

        for row in rows:
            cols = st.columns(COLS)
            for col, film in zip(cols, row):
                with col:
                    poster_url = film.get('poster')
                    name       = film.get('name', '')
                    rating     = film.get('rating')
                    stars      = '★' * int(rating) + ('½' if rating % 1 >= 0.5 else '') if rating else ''
                    short_name = name[:16] + '...' if len(name) > 16 else name

                    if poster_url:
                        st.markdown(f"""
                        <div style="text-align:center; margin-bottom:4px;">
                            <img src="{poster_url}"
                                 style="width:100%; border-radius:6px;
                                        border:1px solid #1e1e2e;"
                                 onerror="this.style.display='none'"/>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="width:100%; aspect-ratio:2/3; background:#111118;
                                    border:1px solid #1e1e2e; border-radius:6px;
                                    display:flex; align-items:center; justify-content:center;
                                    margin-bottom:4px;">
                            <span style="color:#333344; font-size:24px;">🎬</span>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="text-align:center; font-family:'DM Mono',monospace;
                                font-size:9px; color:#666680; line-height:1.4; margin-bottom:4px;">
                        {short_name}<br>
                        <span style="color:#e8c96a;">{stars if stars else '—'}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("▸ Details", key=f"film_{selected_list}_{name}", use_container_width=True):
                        if st.session_state['selected_film'] == name:
                            st.session_state['selected_film'] = None
                        else:
                            st.session_state['selected_film'] = name
                        st.rerun()

        # Detail panel — appears below the grid when a film is selected
        selected = st.session_state.get('selected_film')
        if selected:
            film_detail = next((f for f in films if f['name'] == selected), None)
            if film_detail:
                st.divider()
                rating  = film_detail.get('rating')
                stars   = '★' * int(rating) + ('½' if rating % 1 >= 0.5 else '') if rating else ''
                genres  = film_detail.get('genres', '')
                review  = film_detail.get('review')
                year    = film_detail.get('year', '')
                poster  = film_detail.get('poster')

                col_img, col_info = st.columns([1, 3])
                with col_img:
                    if poster:
                        st.markdown(f"""
                        <img src="{poster}" style="width:100%; border-radius:8px;
                                   border:1px solid #1e1e2e;"/>
                        """, unsafe_allow_html=True)

                with col_info:
                    st.markdown(f"""
                    <div style="padding: 8px 0;">
                        <div style="font-family:'Playfair Display',serif; font-size:24px;
                                    font-weight:700; color:#e8e8f0; margin-bottom:4px;">
                            {selected}
                        </div>
                        <div style="font-family:'DM Mono',monospace; font-size:11px;
                                    color:#444460; margin-bottom:8px;">
                            {year}{' · ' + genres if genres else ''}
                        </div>
                        <div style="font-size:20px; color:#e8c96a; margin-bottom:12px;">
                            {stars if stars else 'Not rated'}
                            {'<span style="font-family:DM Mono,monospace; font-size:11px; color:#444460;"> (' + str(rating) + ')</span>' if rating else ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if review:
                        preview = review
                        st.markdown(f"""
                        <div style="font-size:13px; color:#9999b8; line-height:1.7;
                                    font-style:italic; border-left:2px solid #1e1e2e;
                                    padding-left:16px; margin-top:8px;">
                            "{preview}"
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.caption("No review written for this film.")

                    if st.button("✕  Close", key="close_detail"):
                        st.session_state['selected_film'] = None
                        st.rerun()

# ─────────────────────────────────────────────────────────────
# SECTION 9 — DIRECTOR DEEP-DIVE
# ─────────────────────────────────────────────────────────────
elif section == "Directors":
    st.markdown("""
    <div class="section-title">Director<br><em>Deep Dive</em></div>
    <div class="section-sub">✦ DIRECTORS // PHOTO · BIO · FILMS · RATING OVER TIME</div>
    """, unsafe_allow_html=True)

    directors_deep = data.get('directors_deep', {})

    if not directors_deep:
        st.info("Run scripts 11c_fetch_directors.py and 11_export_data.py first.")
    else:
        # Sort by avg_rating desc for the selector
        dir_names = sorted(
            directors_deep.keys(),
            key=lambda x: directors_deep[x].get('avg_rating', 0),
            reverse=True
        )

        selected_dir = st.selectbox("Select a director", dir_names)
        d = directors_deep[selected_dir]

        st.divider()

        # ── Header: photo + bio ──
        col_photo, col_bio = st.columns([1, 3])

        with col_photo:
            if d.get('photo'):
                st.markdown(f"""
                <img src="{d['photo']}"
                     style="width:100%; border-radius:10px; border:1px solid #1e1e2e;"/>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="width:100%; aspect-ratio:3/4; background:#111118;
                            border:1px solid #1e1e2e; border-radius:10px;
                            display:flex; align-items:center; justify-content:center;">
                    <span style="font-size:40px;">🎬</span>
                </div>
                """, unsafe_allow_html=True)

        with col_bio:
            avg  = d.get('avg_rating', 0)
            cnt  = d.get('count', len(d.get('films', [])))
            birth = d.get('birthday', '')
            place = d.get('birthplace', '')
            stars = '★' * int(avg) + ('½' if avg % 1 >= 0.5 else '') if avg else ''

            st.markdown(f"""
            <div style="padding: 8px 0;">
                <div style="font-family:'Playfair Display',serif; font-size:28px;
                            font-weight:900; color:#e8e8f0; margin-bottom:6px;">
                    {selected_dir}
                </div>
                <div style="font-family:'DM Mono',monospace; font-size:10px;
                            color:#444460; letter-spacing:0.1em; margin-bottom:12px;">
                    {f'{birth[:4]} · {place}' if birth and place else place or birth or ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Films watched", cnt)
            m2.metric("Avg rating", f"{avg:.2f}")
            m3.metric("vs my avg", f"{avg - data['stats']['avg_rating']:+.2f}")

            if d.get('bio'):
                st.markdown(f"""
                <div style="font-size:13px; color:#666680; line-height:1.65;
                            margin-top:12px; font-style:italic;">
                    "{d['bio']}"
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ── Films poster grid ──
        films = d.get('films', [])
        if films:
            st.markdown("""
            <div style="font-family:'DM Mono',monospace; font-size:10px;
                        color:#444460; letter-spacing:0.12em; margin-bottom:12px;">
                FILMS I WATCHED
            </div>""", unsafe_allow_html=True)

            COLS = 6
            rows = [films[i:i+COLS] for i in range(0, len(films), COLS)]

            if 'dir_selected_film' not in st.session_state:
                st.session_state['dir_selected_film'] = None

            for row in rows:
                cols = st.columns(COLS)
                for col, film in zip(cols, row):
                    with col:
                        poster = film.get('poster')
                        name   = film.get('name', '')
                        rating = film.get('rating')
                        stars  = '★' * int(rating) + ('½' if rating % 1 >= 0.5 else '') if rating else ''
                        short  = name[:16] + '...' if len(name) > 16 else name

                        if poster:
                            st.markdown(f"""
                            <div style="text-align:center; margin-bottom:4px;">
                                <img src="{poster}"
                                     style="width:100%; border-radius:6px;
                                            border:1px solid #1e1e2e;"/>
                            </div>""", unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="width:100%; aspect-ratio:2/3; background:#111118;
                                        border:1px solid #1e1e2e; border-radius:6px;
                                        display:flex; align-items:center; justify-content:center;
                                        margin-bottom:4px;">
                                <span style="color:#333344; font-size:20px;">🎬</span>
                            </div>""", unsafe_allow_html=True)

                        st.markdown(f"""
                        <div style="text-align:center; font-family:'DM Mono',monospace;
                                    font-size:9px; color:#666680; line-height:1.4; margin-bottom:4px;">
                            {short}<br>
                            <span style="color:#e8c96a;">{stars if stars else '—'}</span>
                        </div>""", unsafe_allow_html=True)

                        if st.button("▸  details", key=f"dir_{selected_dir}_{name}",
                                     use_container_width=True):
                            if st.session_state['dir_selected_film'] == name:
                                st.session_state['dir_selected_film'] = None
                            else:
                                st.session_state['dir_selected_film'] = name
                            st.rerun()

            # Film detail panel
            sel_film = st.session_state.get('dir_selected_film')
            if sel_film:
                film_detail = next((f for f in films if f['name'] == sel_film), None)
                if film_detail:
                    st.divider()
                    rating  = film_detail.get('rating')
                    stars   = '★' * int(rating) + ('½' if rating % 1 >= 0.5 else '') if rating else ''
                    review  = film_detail.get('review')
                    genres  = film_detail.get('genres', '')
                    year    = film_detail.get('year', '')
                    poster  = film_detail.get('poster')
                    tmdb    = film_detail.get('tmdb_rating')

                    col_img, col_info = st.columns([1, 3])
                    with col_img:
                        if poster:
                            st.markdown(f"""
                            <img src="{poster}" style="width:100%; border-radius:8px;
                                       border:1px solid #1e1e2e;"/>
                            """, unsafe_allow_html=True)

                    with col_info:
                        st.markdown(f"""
                        <div style="padding:8px 0;">
                            <div style="font-family:'Playfair Display',serif; font-size:22px;
                                        font-weight:700; color:#e8e8f0; margin-bottom:4px;">
                                {sel_film}
                            </div>
                            <div style="font-family:'DM Mono',monospace; font-size:10px;
                                        color:#444460; margin-bottom:8px;">
                                {year}{' · ' + genres if genres else ''}
                            </div>
                            <div style="font-size:18px; color:#e8c96a; margin-bottom:8px;">
                                {stars if stars else 'Not rated'}
                                {'<span style="font-family:DM Mono,monospace; font-size:10px; color:#444460;"> my rating · TMDb ' + str(round(tmdb, 1)) + '</span>' if tmdb else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if review:
                            st.markdown(f"""
                            <div style="font-size:13px; color:#9999b8; line-height:1.7;
                                        font-style:italic; border-left:2px solid #1e1e2e;
                                        padding-left:16px; margin-top:8px;">
                                "{review}"
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.caption("No review written for this film.")

                        if st.button("✕  Close", key="close_dir_detail"):
                            st.session_state['dir_selected_film'] = None
                            st.rerun()

        st.divider()

        # ── Rating over time ──
        rot = d.get('rating_over_time', [])
        if len(rot) >= 2:
            st.markdown("""
            <div style="font-family:'DM Mono',monospace; font-size:10px;
                        color:#444460; letter-spacing:0.12em; margin-bottom:12px;">
                RATING OVER TIME
            </div>""", unsafe_allow_html=True)

            rot_df = pd.DataFrame(rot)
            rot_df['date'] = pd.to_datetime(rot_df['date'])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rot_df['date'], y=rot_df['rating'],
                mode='markers+lines',
                marker=dict(color=GOLD, size=8),
                line=dict(color=GOLD, width=1.5),
                text=rot_df['name'],
                hovertemplate='<b>%{text}</b><br>%{x|%Y}<br>★ %{y}<extra></extra>'
            ))
            fig.add_hline(y=data['stats']['avg_rating'], line_dash='dash',
                          line_color='#333344',
                          annotation_text=f"my avg ({data['stats']['avg_rating']})",
                          annotation_font_color='#444460')
            fig.update_layout(**plotly_layout(
                title=f'When I watched {selected_dir} — and how I rated it',
                height=300, showlegend=False
            ))
            fig.update_xaxes(title=None)
            fig.update_yaxes(range=[0, 5.5], title=None)
            st.plotly_chart(fig, use_container_width=True)

        # ── Watchlist ──
        watchlist_films = d.get('watchlist', [])
        if watchlist_films:
            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace; font-size:10px;
                        color:#444460; letter-spacing:0.12em; margin:12px 0;">
                STILL TO WATCH ({len(watchlist_films)} films on watchlist)
            </div>""", unsafe_allow_html=True)

            wl_cols = st.columns(min(len(watchlist_films), 8))
            for col, film in zip(wl_cols, watchlist_films):
                with col:
                    poster = film.get('poster')
                    name   = film.get('name', '')
                    short  = name[:14] + '...' if len(name) > 14 else name
                    if poster:
                        st.markdown(f"""
                        <div style="text-align:center;">
                            <img src="{poster}" style="width:100%; border-radius:6px;
                                       border:1px solid #1e1e2e; margin-bottom:4px;"/>
                            <div style="font-family:'DM Mono',monospace; font-size:8px;
                                        color:#444460;">{short}</div>
                        </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Word Cloud of reviews ──
        all_reviews_text = ' '.join([
            f.get('review', '') or ''
            for f in films
            if f.get('review')
        ])

        if all_reviews_text.strip():
            st.markdown("""
            <div style="font-family:'DM Mono',monospace; font-size:10px;
                        color:#444460; letter-spacing:0.12em; margin-bottom:12px;">
                WORD CLOUD — MY REVIEWS ABOUT THIS DIRECTOR
            </div>""", unsafe_allow_html=True)

            try:
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                import io as _io

                # Custom stopwords
                stopwords = {
                    'the','a','an','and','or','but','in','on','at','to','for',
                    'of','with','is','it','as','this','that','was','are','be',
                    'have','has','had','not','do','did','i','my','me','he','she',
                    'his','her','its','we','they','their','our','film','movie',
                    'one','just','more','also','so','very','much','even','from',
                    'um','uma','que','de','em','para','com','não','se','por',
                    'o','a','os','as','e','é','no','na','ao','da','do','dos',
                    'das','mas','seu','sua','ele','ela','eles','isso','este'
                }

                wc = WordCloud(
                    width=900, height=400,
                    background_color='#111118',
                    colormap='YlOrRd',
                    stopwords=stopwords,
                    max_words=80,
                    prefer_horizontal=0.8,
                    min_font_size=10,
                    font_step=1,
                    random_state=42,
                ).generate(all_reviews_text)

                fig, ax = plt.subplots(figsize=(12, 5))
                fig.patch.set_facecolor('#111118')
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout(pad=0)

                buf = _io.BytesIO()
                plt.savefig(buf, format='png', dpi=150,
                            bbox_inches='tight', facecolor='#111118')
                buf.seek(0)
                plt.close()
                st.image(buf, use_container_width=True)

            except Exception as e:
                st.warning(f"Could not generate word cloud: {e}")
        else:
            st.info("No reviews written for this director's films.")

        st.divider()

        # ── Compare two directors ──
        st.markdown("""
        <div style="font-family:'DM Mono',monospace; font-size:10px;
                    color:#444460; letter-spacing:0.12em; margin-bottom:12px;">
            COMPARE WITH ANOTHER DIRECTOR
        </div>""", unsafe_allow_html=True)

        other_dirs = [d for d in dir_names if d != selected_dir]
        compare_dir = st.selectbox("Compare with", other_dirs, key="compare_dir_select")

        if compare_dir and compare_dir in directors_deep:
            d2 = directors_deep[compare_dir]
            films2 = d2.get('films', [])

            col_a, col_mid, col_b = st.columns([2, 1, 2])

            def dir_stats_block(director_name, director_data, director_films, align='left'):
                avg    = director_data.get('avg_rating', 0)
                cnt    = len(director_films)
                rated  = [f['rating'] for f in director_films if f.get('rating')]
                best   = max(director_films, key=lambda f: f.get('rating') or 0, default={})
                worst  = min(director_films, key=lambda f: f.get('rating') or 5, default={})

                text_align = 'right' if align == 'left' else 'left'
                return f"""
                <div style="text-align:{text_align};">
                    <div style="font-family:'Playfair Display',serif; font-size:20px;
                                font-weight:700; color:#e8e8f0; margin-bottom:8px;">
                        {director_name}
                    </div>
                    <div style="font-family:'DM Mono',monospace; font-size:11px;
                                color:#444460; margin-bottom:4px;">
                        {cnt} films watched
                    </div>
                    <div style="font-size:22px; color:#e8c96a; margin-bottom:8px;">
                        {'★' * int(avg)}{'½' if avg % 1 >= 0.5 else ''}
                        <span style="font-size:13px; color:#666680;"> {avg:.2f}</span>
                    </div>
                    <div style="font-family:'DM Mono',monospace; font-size:10px; color:#666680;">
                        BEST: {best.get('name','—')} ({'★'*int(best.get('rating',0)) if best.get('rating') else '—'})<br>
                        WORST: {worst.get('name','—')} ({'★'*int(worst.get('rating',0)) if worst.get('rating') else '—'})
                    </div>
                </div>"""

            with col_a:
                st.markdown(dir_stats_block(selected_dir, d, films, 'left'),
                            unsafe_allow_html=True)
                if d.get('photo'):
                    st.markdown(f"""
                    <div style="text-align:right; margin-top:12px;">
                        <img src="{d['photo']}" style="width:80%; border-radius:8px;
                                   border:1px solid #1e1e2e;"/>
                    </div>""", unsafe_allow_html=True)

            with col_mid:
                avg1 = d.get('avg_rating', 0)
                avg2 = d2.get('avg_rating', 0)
                winner = selected_dir if avg1 > avg2 else compare_dir
                st.markdown(f"""
                <div style="text-align:center; padding-top:20px;">
                    <div style="font-family:'DM Mono',monospace; font-size:9px;
                                color:#333350; letter-spacing:0.1em; margin-bottom:8px;">
                        VS
                    </div>
                    <div style="font-family:'DM Mono',monospace; font-size:9px;
                                color:#444460; margin-top:16px;">
                        I prefer<br>
                        <span style="color:#e8c96a; font-size:11px;">{winner}</span>
                    </div>
                    <div style="font-family:'DM Mono',monospace; font-size:9px;
                                color:#333350; margin-top:8px;">
                        by {abs(avg1-avg2):.2f}★
                    </div>
                </div>""", unsafe_allow_html=True)

            with col_b:
                st.markdown(dir_stats_block(compare_dir, d2, films2, 'right'),
                            unsafe_allow_html=True)
                if d2.get('photo'):
                    st.markdown(f"""
                    <div style="text-align:left; margin-top:12px;">
                        <img src="{d2['photo']}" style="width:80%; border-radius:8px;
                                   border:1px solid #1e1e2e;"/>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # Rating comparison chart
            ratings1 = sorted([f['rating'] for f in films if f.get('rating')])
            ratings2 = sorted([f['rating'] for f in films2 if f.get('rating')])

            fig = go.Figure()
            fig.add_trace(go.Box(
                y=ratings1, name=selected_dir,
                marker_color=GOLD, line_color=GOLD,
                fillcolor='rgba(232,201,106,0.15)',
            ))
            fig.add_trace(go.Box(
                y=ratings2, name=compare_dir,
                marker_color=TEAL, line_color=TEAL,
                fillcolor='rgba(0,204,170,0.15)',
            ))
            fig.update_layout(**plotly_layout(
                title='Rating distribution comparison',
                height=320, showlegend=True
            ))
            fig.update_yaxes(range=[0, 5.5], title='Rating')
            st.plotly_chart(fig, use_container_width=True)