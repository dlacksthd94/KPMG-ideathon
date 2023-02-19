#!/usr/bin/python
# -*- coding: <encoding name> -*-

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
import pickle
from graph import get_plotly_graph, filtering_dart_graph, processing_keywords
from news import get_articles, get_accodion_items
import json
import requests
from bs4 import BeautifulSoup as bs
import networkx as nx
import time

# app = Dash(__name__)
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
G_loaded = nx.read_gml(path='/home/kic/KPMG-ideathon/KG/dart_graph')


### 연관 기사 파트 ### 


### 기업 정보 파트 ###
path = '/home/kic/data'
pickle_file_1 = f'{path}/dart_processed/summ_EV.pickle'
pickle_file_2 = f'{path}/dart_processed/html_EV.pickle'
# Dart Summarization data
with open(pickle_file_1, 'rb') as f:
    docs = pickle.load(f)
# Dart Summ + NER data
with open(pickle_file_2, 'rb') as f:
    ners = pickle.load(f)
EV = pd.read_csv(f'{path}/kodex/EV_processed.csv', dtype=object)
name = EV.corp_code


## News 읽기
df_cs = pd.read_pickle('~/data/df_news_final.pickle')
df_cs['date'] = pd.to_datetime(df_cs['date'])

titles, contents, dates, urls = get_articles(df_cs)
### 그래프 파트 ### 
fig = get_plotly_graph(df_cs)

min_date = df_cs['date'].min()
max_date = df_cs['date'].max()

daterange = pd.date_range(start=min_date,end=max_date,freq='D')
def unixTimeMillis(dt):
    ''' Convert datetime to unix timestamp '''
    return int(time.mktime(dt.timetuple()))

def unixToDatetime(unix):
    ''' Convert unix timestamp to datetime. '''
    # return str(date.strftime('%Y-%m-%d'))
    return pd.to_datetime(unix,unit='s').strftime('%Y-%m-%d')

def getMarks(start, end, Nth=14):
    ''' Returns the marks for labeling. 
        Every Nth value will be used.
    '''

    result = {}
    for i, date in enumerate(daterange):
        if(i%Nth == 1):
            # Append value to dict
            result[unixTimeMillis(date)] = str(date.strftime('%Y-%m-%d'))
    return result

range_slider = dcc.RangeSlider(
                id='year_slider',
                min = unixTimeMillis(daterange.min()),
                max = unixTimeMillis(daterange.max()),
                value = [unixTimeMillis(daterange.min()),
                         unixTimeMillis(daterange.max())],
                marks=getMarks(daterange.min(),
                            daterange.max()),
            )

# txt = docs[name[61]]['summ'][0].replace('1. 사업의 개요 ', '')
# ner_1 = ners[name[61]]['ner'][1]
# ner_2 = ners[name[61]]['ner'][2]

# base_url = 'https://dart.fss.or.kr'
# url = f'{base_url}/navi/searchNavi.do?naviCrpCik={name[61]}&naviCode=A002'
# response = requests.get(url=url)
# soup = bs(response.text, "html.parser")
# new_url = f"{base_url}{soup.find('iframe')['src']}"


### 앱 레이아웃 ### 
app.layout = dbc.Container([

    # Header
    html.Div(
        children=[
            html.Img(src='https://github.com/dlacksthd94/KPMG-ideathon/blob/main/app/header.png?raw=true',
                    style={'height':'40%', 'width':'40%', 'margin':30}),
            html.Div(
                children = [
                    html.Div(id='year_filter_text', style={'text-align': 'center'}),
                    range_slider
                ],
                style={'height':'5%', 'width':'40%', 'margin':30, 'font-size':'80%'}
           ),

        ]
    ),
    

    html.Div(
        children=[
            # Col 1 (연관 기사)
            html.Div(
                style={'width':'28%', 'height':'120%','float':'left', 'margin':0, 'overflow':'scroll', "maxHeight": "420px"},

                children=[],
                id = 'accodion'
            ),


            # Col 2 (지식 그래프)
            html.Div(
                style={'width':'39%', 'height':'150%','float':'left', 'margin':5},
                children=[
                    # html.P('지식 그래프'),
                    dcc.Graph(
                        id='graph1',
                        figure=fig
                    )
                ],
            ),


            # Col 3 (기업 정보 요약)
            html.Div(
                style={'width':'30%', 'height':'100%','float':'left', 'margin':'0 0 0 30px'},
                children=[
                    html.Div(
                        children=[
                            html.H5(
                                children='관심 있는 노드를 클릭해보세요.', 
                                id='company_name'
                            ),
                            # html.Br(),
                            
                            html.Div(
                                children=[
                                    html.H6("사업 개요"),
                                    html.Div(id='company_view'),

                                    html.Div(
                                        children=[
                                            dbc.Button(
                                                # "출처: 금융감독원 전자공시시스템", 
                                                "사업보고서 원문",
                                                id="open-offcanvas", 
                                                n_clicks=0, 
                                                size='sm',
                                                color='dark',
                                                outline=True,
                                                style={'width':'25%', 'height':'50%', 'font-size':'70%'}
                                            ),
                                            dbc.Offcanvas(
                                                id="offcanvas",
                                                placement="end",
                                                is_open=False,
                                                style={'width':'50%'},
                                                children=[
                                                    dbc.Spinner(
                                                        children=[
                                                            html.Iframe(id="iframe", src="about:blank", style={"width": "100%", "height": "calc(100vh - 50px)"})
                                                        ], 
                                                        type="grow"
                                                    ),
                                                ],
                                            )
                                        ],
                                        style={'textAlign':'right', 'font-size': '90%'}
                                    ),   
                                ]
                            ),
                            html.Br(),  
                            html.Div(
                                children=html.H6(
                                    id='dart_result'
                                ),
                            ),
                        ],
                        style={'width':'100%', 'height':'80%'}
                    ),         
                ]
            )
        ]
    )
], fluid=True)



@app.callback(
    Output('graph1', 'children'),
    Input('graph1', 'selectedData'))
def display_selected_data(selectedData):
    print(json.dumps(selectedData, indent=2))
    return json.dumps(selectedData, indent=2)
def dart_company_view(c_code):
    # print(docs[c_code])
    txt = docs[c_code]['summ'][0].replace('1. 사업의 개요 ', '')
    return html.Div(children=txt)

@app.callback(
    Output('company_name', 'children'),
    Output('company_view', 'children'),
    Input('graph1', 'selectedData'))
def display_selected_data(selectedData):
    global c_code, new_url
    c_name = selectedData["points"][0]["text"]
    c_row = EV[EV['corp_name'] == c_name]
    if len(c_row) > 0:
        c_code = c_row['corp_code'].values[0]

        base_url = 'https://dart.fss.or.kr'
        url = f'{base_url}/navi/searchNavi.do?naviCrpCik={c_code}&naviCode=A002'
        response = requests.get(url=url)
        soup = bs(response.text, "html.parser")
        new_url = f"{base_url}{soup.find('iframe')['src']}"
        return f'{c_name}', dart_company_view(c_code)

    else:
        url = f"https://ko.wikipedia.org/wiki/{c_name.replace(' ', '+')}"
        iframe_view = html.Iframe(src=url, style={"width": "100%", "height": "calc(100vh - 50px)"})
        return f'{c_name}', iframe_view
        

@app.callback(
    Output('dart_result', 'children'),
    Input('graph1', 'selectedData'))
def display_selected_data(selectedData):
    corp_name = selectedData['points'][0]['text']
    keywords = filtering_dart_graph(G_loaded, corp_name)
    childrens = [
        html.H6(f"{corp_name} 관련 기업 및 제품 키워드"),
        html.P(", ".join(keywords['주요제품1']), style={'font-size':'80%'}),
        # html.H6(f"{corp_name} 관련 기업"),
        # html.P(", ".join(keywords['관계사']), style={'font-size':'80%'}),
        # html.H6(f"{corp_name} 관련 원자재"),
        # html.P(", ".join(keywords['원재료']), style={'font-size':'80%'}),
    ]
    return childrens


    
@app.callback(
    Output("offcanvas", "is_open"),
    Output("iframe", "src"),
    Input("open-offcanvas", "n_clicks"),
    prevent_initial_call=True
)
def open_offcanvas(n_clicks):
    if n_clicks:
        return True, new_url
    return False, "about:blank"

@app.callback(
    Output('year_filter_text', 'children'),
    Output('graph1', 'figure'),
    [Input('year_slider', 'value')])
def _update_time_range_label(year_range):
    start_date = unixToDatetime(year_range[0])
    end_date = unixToDatetime(year_range[1])
    new_df = df_cs[(df_cs['date'] >= start_date) & (df_cs['date'] <= end_date)]
    filter_text_string = '날짜 필터: {} 부터 {} 까지'.format( start_date, end_date )
    fig = get_plotly_graph(new_df)
    return filter_text_string, fig

@app.callback(
    Output('accodion', 'children'),
    [Input('year_slider', 'value')])
def _update_time_range_label(year_range):
    start_date = unixToDatetime(year_range[0])
    end_date = unixToDatetime(year_range[1])
    new_df = df_cs[(df_cs['date'] >= start_date) & (df_cs['date'] <= end_date)]  
    return get_accodion_items(new_df)

if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=False, host='localhost', port=8050, use_reloader=True)