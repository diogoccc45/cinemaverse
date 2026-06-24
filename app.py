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
    st.markdown("""
    <div class="section-title">768 films.<br><em>A life</em> watching.</div>
    <div class="section-sub">◈ OVERVIEW // EVERY FILM I'VE WATCHED — IN NUMBERS</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Films",      s['total_watched'])
    c2.metric("Avg rating", s['avg_rating'])
    c3.metric("5-star",     s['five_star_count'])
    c4.metric("0.5-star",   s['half_star_count'])
    c5.metric("Reviews",    s['total_reviews'])
    c6.metric("Watchlist",  s['watchlist_count'])
    c7.metric("Years",      f"{s['first_log'][:4]}–{s['last_log'][:4]}")

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

    if not reviews_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            min_rating = st.slider("Min rating", 0.5, 5.0, 3.5, 0.5)
        with col2:
            min_length = st.slider("Min length (chars)", 0, 500, 80, 40)
        with col3:
            search_r = st.text_input("", placeholder="Search in text...", label_visibility="collapsed")

        filtered = reviews_df.copy()
        if 'Rating' in filtered.columns:
            filtered = filtered[filtered['Rating'] >= min_rating]
        filtered = filtered[filtered['review_length'] >= min_length]
        if search_r:
            filtered = filtered[filtered['Review'].str.contains(search_r, case=False, na=False)]

        st.caption(f"{len(filtered)} reviews match")

        for _, row in filtered.head(12).iterrows():
            rating = row.get('Rating')
            stars  = '★' * int(rating) + ('½' if rating % 1 >= 0.5 else '') if pd.notna(rating) else ''
            preview = str(row['Review'])[:350] + ('...' if len(str(row['Review'])) > 350 else '')
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
    else:
        st.info("No reviews data found.")