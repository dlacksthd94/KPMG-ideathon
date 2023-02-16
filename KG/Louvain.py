
# coding: utf-8

# In[1]:

import pandas as pd
import networkx as nx
import itertools
import community as community_louvain
import json
from pyvis.network import Network


# ## Louvain graph 함수 (threshold 적용)

# In[2]:

def louvain(tlcc_df, lower_threshold, upper_threshold, cluster_result_path, var_group_dict_path):
    
    # TLCC Matrix 값들에 절대값 적용
    dist_matrix = tlcc_df.abs()
    
    # 네트워크 생성    
    df_edges = pd.DataFrame(index=range(len(list(itertools.combinations(dist_matrix.columns, 2)))),columns=range(3))
    df_edges.columns = ['col1','col2','weight']
    cols = dist_matrix.columns
    for i,j in enumerate(itertools.combinations(cols, 2)):
        df_edges.iloc[i,0] = j[0]
        df_edges.iloc[i,1] = j[1]
        df_edges.iloc[i,2] = dist_matrix.loc[j[0],j[1]]
        
    df_edges_new = df_edges[(df_edges['weight']>=lower_threshold) & (df_edges['weight']<= upper_threshold)]
    
    edges = df_edges_new[['col1','col2']].values.tolist()
    weights = [float(l) for l in df_edges_new.weight.values.tolist()]

    G = nx.Graph(directed=True)
    G.add_edges_from(edges)
    for cnt, a in enumerate(G.edges(data=True)):
        G.edges[(a[0],a[1])]['weight']=weights[cnt]
        
    # 르방 알고리즘으로 클러스터링된 dictionary 생성        
    partition = community_louvain.best_partition(G, weight='weight')
    louvain_cluster_dict = dict()
    

    for key,value in partition.items():
        if value not in louvain_cluster_dict:
            louvain_cluster_dict[value]=[key]
        else:
            louvain_cluster_dict[value].append(key)
    
    # 클러스터 결과를 데이터프레임으로 생성 및 csv로 저장
    def cluster_result(cluster_dict):
        cluster_result_df = pd.DataFrame()
        for key,value in cluster_dict.items():
            cluster_i = pd.Series(value, name="Cluster"+str(key))
            cluster_result_df = pd.concat([cluster_result_df,cluster_i],axis=1)
        return cluster_result_df
    
    cluster_result_df = cluster_result(louvain_cluster_dict)
    cluster_result_df.to_csv(cluster_result_path,index=False,encoding='utf-8-sig')

    # 클러스터 결과를 네트워크 시각화에 input할 수 있는 형태로 저장
    var_group_dict = dict()
    for key, val in louvain_cluster_dict.items():
        for i in val:
            var_group_dict[i] = str(key)
    
    with open(var_group_dict_path,'w',encoding='UTF-8-sig') as f:
        json.dump(var_group_dict,f, ensure_ascii=False)
            
    return cluster_result_df, var_group_dict


# ## 네트워크 그래프 시각화 함수

# In[3]:

def draw_graph(tlcc_df, time_lag_df, p_value_df, lower_threshold, upper_threshold, var_group_dict, vis_result_path):
     # 네트워크 생성 함수
    def network_graph(dist_matrix,lower_threshold,upper_threshold):
        df_edges = pd.DataFrame(index=range(len(list(itertools.combinations(dist_matrix.columns, 2)))),columns=range(3))
        df_edges.columns = ['col1','col2','weight']
        cols = dist_matrix.columns
        for i,j in enumerate(itertools.combinations(cols, 2)):
            df_edges.iloc[i,0] = j[0]
            df_edges.iloc[i,1] = j[1]
            df_edges.iloc[i,2] = dist_matrix.loc[j[0],j[1]]

        df_edges_new = df_edges[(df_edges['weight']>=lower_threshold) & (df_edges['weight']<= upper_threshold)]

        edges = df_edges_new[['col1','col2']].values.tolist()
        weights = [float(l) for l in df_edges_new.weight.values.tolist()]

        G = nx.Graph(directed=True)
        G.add_edges_from(edges)
        for cnt, a in enumerate(G.edges(data=True)):
            G.edges[(a[0],a[1])]['weight']=weights[cnt]
        return G

    # TLCC Matrix 값들에 절대값 적용
    dist_matrix = tlcc_df.abs()
    
    # 네트워크 생성 함수 실행
    G = network_graph(dist_matrix, lower_threshold, upper_threshold)
    
    # 네트워크 시각화
    net = Network(notebook=True, height="700px", width='100%')

    net.from_nx(G)
    net.repulsion(node_distance=50, central_gravity=0, spring_length=200, spring_strength=0.05, damping=0.09)
    net.toggle_hide_edges_on_drag(True)

    net.set_options
    net.show_buttons() # filter_=['physics']

    neighbor_map = net.get_adj_list()

    # 노드 정보 지정
    for node in net.nodes:
        # 노드를 클릭하였을 때 이웃 노드의 정보를 보여주기
        node['title'] = node['id'] + '<br>(Neighbor, Correlation, Days to shift, P-value):<br><br>'
        for i in neighbor_map[node['id']]:
            corr = str(round(tlcc_df.loc[node['id'], i],2))
            lag = str(time_lag_df.loc[node['id'], i])
            pvalue = str(round(p_value_df.loc[node['id'], i],4))
            node['title'] += i + ', ' + corr + ', ' + lag + ' Days, '+ pvalue + '<br>'
        # 노드의 크기는 연결된 이웃의 개수에 비례
        node['value'] = len(neighbor_map[node['id']])

    # 르방 알고리즘 클러스터링 결과에 따라 노드 색 지정
    for node in net.nodes:
        node['group'] = str(var_group_dict[node['id']]) 
    # 엣지를 클릭하였을 때 두 노드 간에 정보 보여주기
    for edge in net.edges:
        corr = str(round(tlcc_df.loc[edge['from'], edge['to']],2))
        days = str(time_lag_df.loc[edge['from'], edge['to']])
        pvalue = str(round(p_value_df.loc[edge['from'], edge['to']],4))
        edge['title'] = 'Corr :' + corr + ' (p-value : ' + pvalue + '),<br>Time-lag : shift "' + edge['to'] + '" '+ days + ' days'

    net.show(vis_result_path)
    
    return net



