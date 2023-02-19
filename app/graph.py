import pandas as pd
import networkx as nx
import itertools
import community as community_louvain
import json
from pyvis.network import Network
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import math
from collections import deque


def update_point(trace, points, selector):
    print("Clicked", trace, points, selector)

def get_top_n_nodes(n, nodes, edges):
    sorted_nodes = {k: v for k, v in sorted(nodes.items(), key=lambda item: item[1], reverse=True)}

    result_nodes = {}
    remove_list = []
    for i, (k,v) in enumerate(sorted_nodes.items()):
        if i < n:
            result_nodes[k] = v
        else:
            for keys in edges.keys():
                if k in keys:
                    remove_list.append(keys)
    result_edges = {k:v for k,v in edges.items() if k not in remove_list}

    return result_nodes, result_edges

def retrieve_news_graph(df_cs):
    global nodes_info
    if df_cs is None:
        df_cs = pd.read_pickle('~/data/df_news_final.pickle')
    result = []
    nodes = dict()
    edges = dict()
    nodes_info = dict()
    
    remove_list = [
        '원장', '과장', '관계자', '사장', '상무', '고객', '챗GPT', '드론', '핵무기'
    ]

    replace_dict = {
        '폐배터': '폐배터리',
        '배터': '배터리',
        '이차전지': '2차전지',
        # '테스': '테슬라'
    }

    for _, row in df_cs.iterrows():
        ner_list = row['ner']
        for ners in ner_list:
            # Loop through ner within sentences
            for i in range(len(ners)):
                pair_1 = ners[i][0]

                if pair_1 in remove_list:
                    continue
                if pair_1 in replace_dict:
                    pair_1 = replace_dict[pair_1]

                if pair_1 in nodes:
                    nodes[pair_1] += 1
                else:
                    nodes[pair_1] = 1
                nodes_info[pair_1] = ners[i][1]

                for j in range(i+1, len(ners)):
                    pair_2 = ners[j][0]

                    if pair_2 in remove_list:
                        continue
                    if pair_2 in replace_dict:
                        pair_2 = replace_dict[pair_2]


                    if pair_1 == pair_2:
                        continue
                    key = frozenset((pair_1, pair_2))
                    if key in edges:
                        edges[key] += 1
                    else:
                        edges[key] = 1
    result = get_top_n_nodes(50, nodes, edges)

    node_size = [ int(math.log(x) *3 ) for x in result[0].values()]

    color_dict = {
        'ARTIFACT': 'rgba(177, 198, 234, 1)', 
        'CIVILIZATION': 'rgba(177, 198, 234, 1)', 
        'ORGANIZATION': 'rgba(54, 61, 89, 1)', 
        'PERSON': 'rgba(177, 198, 234, 1)', 
        'TERM': 'rgba(247, 230, 216, 1)',
    }
    color_map = [
        color_dict[nodes_info[k]] if nodes_info[k] in color_dict else 'lightgray'
        for k in result[0].keys()
    ]


    edge_width = [ 0.1 if v < 20 else math.log(v) for v in result[1].values()]

    G=nx.Graph()
    G.add_nodes_from(result[0].keys())
    G.add_edges_from(result[1].keys())
    return G, node_size, color_map, edge_width

def retrieve_dart_graph():

    data1_path = '/home/kic/data/dart_processed/dict_sector_product_processed.pickle'
    data2_path = '/home/kic/data/dart_processed/html_EV_v2.pickle'
    ev_list_path = '/home/kic/data/kodex/EV_processed.csv'

    with open(data1_path, 'rb') as f:
        data1 = pickle.load(f)

    with open(data2_path, 'rb') as f:
        data2 = pickle.load(f)

    ev_list = pd.read_csv(ev_list_path, dtype=object)

    g = nx.Graph()

    corp_name = []
    for key in data2.keys():
        corp_name.append(data2[key]['corp_name'])
    g.add_nodes_from(corp_name, kind='corporation')

    주요제품_level1 = []
    for key in data2.keys():
        주요제품_level1 += [product for product in data1[key]]

    g.add_nodes_from(주요제품_level1, kind='주요제품')

    for key in data2.keys():
        주요제품_level1_temp = [product for product in data1[key]]
        회사명 = data2[key]['corp_name']
        edges = [(회사명, product) for product in 주요제품_level1_temp]
        g.add_edges_from(edges, kind='주요제품')


    return g

def get_plotly_graph(df=None):
    G, node_size, color_map, edge_width = retrieve_news_graph(df)
    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        # x0, y0 = G.nodes[edge[0]]['pos']
        # x1, y1 = G.nodes[edge[1]]['pos']
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

    node_x = []
    node_y = []
    node_label = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_label.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        text=node_label,
        mode='markers+text',
        # hoverinfo='text',
        marker=dict(
            color=color_map,
            size=node_size,
        line_width=2))


    node_trace.on_click(update_point)



    # node_adjacencies = []
    # node_text = []
    # for node, adjacencies in enumerate(G.adjacency()):
    #     # print(adjacencies)
    #     node_adjacencies.append(len(adjacencies[1]))
    #     # node_text.append('# of connections: '+str(len(adjacencies[1])))
    #     node_text.append(adjacencies[0])

    # node_trace.marker.color = node_adjacencies
    # node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
        # title='NewsNet',
        titlefont_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        # annotations=[ dict(
        #     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
        #     showarrow=False,
        #     xref="paper", yref="paper",
        #     x=0.005, y=-0.002 ) ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )
    fig.update_traces(textposition='middle center')
    # scatter = fig.data[0]
    # scatter.on_click(update_point)

    fig.update_layout(clickmode='event+select')

    return fig

