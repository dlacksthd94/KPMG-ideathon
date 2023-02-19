#!/usr/bin/python
# -*- coding: <encoding name> -*-

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
import pickle
from graph import get_plotly_graph 
from news import get_articles, get_accodion_items
import json
import requests
from bs4 import BeautifulSoup as bs
import networkx as nx
import time
from datetime import date, datetime
from info import get_info, filtering_dart_graph

# app = Dash(__name__)
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
G_loaded = nx.read_gml(path='/home/kic/KPMG-ideathon/KG/dart_graph')

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

range_slider = dcc.DatePickerRange(
                    id='year_slider',
                    start_date_placeholder_text="Start Period",
                    end_date_placeholder_text="End Period",
                    clearable=True,
                    min_date_allowed=min_date,
                    max_date_allowed=max_date,
                    start_date=min_date,
                    end_date=max_date,
                    # start_date=unixTimeMillis(daterange.min()),
                    # end_date=unixTimeMillis(daterange.max()),
                    style={'width':'100%', 'height':'25px', 'font-size':'20%',  
                    'margin':'10 0 0 10px', 'border-radious':'30px'}
                )




### 앱 레이아웃 ### 
app.layout = dbc.Container([

    # Header
    html.Div(
        children=[
            # Header image
            dbc.Row(
                html.Img(src='https://github.com/dlacksthd94/KPMG-ideathon/blob/main/app/header.png?raw=true',
                    style={'height':'20%', 'width':'20%', 'margin':'-10px 0px 30px 0px'}),
            ),
            
            
            dbc.Row([
                # input box
                dbc.Col(
                    dcc.Input(
                        placeholder=' 뉴스 키워드를 입력하세요...',
                        type='text',
                        value='',
                        style={'width':'100%', 'height':'48px', 'margin':'', 'border-radius': '5px', 'outline':'dark'}
                    ),
                    width=6
                ),
                # 날짜 필터
                dbc.Col(
                    children = [
                        # html.Div(id='year_filter_text', style={'text-align': 'center', 'margin':10}),
                        range_slider
                    ],
                    width=3
                ),
                # 검색 버튼
                dbc.Col(
                    html.Button(
                        '검색', 
                        id='search-button', 
                        style={'margin':'0px 0px 0px 0px', 'width':'100px', 'height':'48px', 'align':'center'},
                    ),
                    width=1
                ),  
                ], justify='center'
            ),

        ],
        style={'margin':30}
    ),
    

    html.Div(
        children=[
            # Col 1 (연관 기사)
            html.Div(
                style={'width':'28%', 'height':'120%','float':'left', 'margin':'10px 0 0 0px', 'overflow':'scroll', "maxHeight": "420px"},

                children=[],
                id = 'accodion'
            ),


            # Col 2 (지식 그래프)
            html.Div(
                style={'width':'39%', 'height':'150%','float':'left', 'margin':'-10px 0 0 0px'},
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
                style={'width':'30%', 'height':'100%','float':'left', 'margin':'24px 0 0 30px'},
                children=[
                    html.Div(
                        children=[
                            html.H5(
                                children='관심 있는 노드를 클릭해보세요.', 
                                id='company_name'
                            ),
                            html.Div(id='company_view', style={'margin':'0 20 0 10px', 'font-size':'90%'}),
                            html.Div(
                                children=[],
                                style={'font-size': '90%'},
                                id='dart_link'
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
    # return html.Div(children=txt)
    return html.Div(children=[html.H6("사업 개요"), txt])

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
        html.P(", ".join(keywords['주요제품1']+keywords['관계사']+keywords['원재료']), style={'font-size':'80%'}),
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
    Output('dart_link', 'children'),
    Input('graph1', 'selectedData'),
    prevent_initial_call=True
)
def display_selected_data(selectedData):
    return get_info()

@app.callback(
    # Output('year_filter_text', 'children'),
    Output('graph1', 'figure'),
    Input('year_slider', 'start_date'),
    Input('year_slider', 'end_date'))

    # [Input('year_slider', 'value')]
    # [Input('year_slider', 'start_date'), 
    # Input('year_slider', 'end_date')])
# def _update_time_range_label(year_range):
def _update_time_range_label(start_date, end_date):
    if start_date is None:
        start_date = min_date

    if end_date is None:
        end_date = max_date

    new_df = df_cs[(df_cs['date'] >= start_date) & (df_cs['date'] <= end_date)]
    filter_text_string = '날짜 필터: {} 부터 {} 까지'.format( start_date, end_date )
    fig = get_plotly_graph(new_df)
    return fig

@app.callback(
    Output('accodion', 'children'),
    # [Input('year_slider', 'value')]
    Input('year_slider', 'start_date'),
    Input('year_slider', 'end_date') )
# def _update_time_range_label(year_range):
def _update_time_range_label(start_date, end_date):
    if start_date is None:
        start_date = min_date

    if end_date is None:
        end_date = max_date    
    new_df = df_cs[(df_cs['date'] >= start_date) & (df_cs['date'] <= end_date)]  
    return get_accodion_items(new_df)

if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=False, host='localhost', port=8080, use_reloader=True)