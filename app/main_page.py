#!/usr/bin/python
# -*- coding: <encoding name> -*-

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
import pickle
from graph import get_plotly_graph
from news import get_articles
import json
import requests
from bs4 import BeautifulSoup as bs

# app = Dash(__name__)
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

### 연관 기사 파트 ### 
titles, contents, dates, urls = get_articles()


### 그래프 파트 ### 
df = px.data.iris() # iris data 불러오기
# fig = px.scatter(df, x="sepal_length", y="sepal_width", color="species") # plotly를 이용한 산점도
fig = get_plotly_graph()




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
txt = docs[name[61]]['summ'][0].replace('1. 사업의 개요 ', '')
ner_1 = ners[name[61]]['ner'][1]
ner_2 = ners[name[61]]['ner'][2]

base_url = 'https://dart.fss.or.kr'
url = f'{base_url}/navi/searchNavi.do?naviCrpCik={name[61]}&naviCode=A002'
response = requests.get(url=url)
soup = bs(response.text, "html.parser")
new_url = f"{base_url}{soup.find('iframe')['src']}"


### 앱 레이아웃 ### 
app.layout = dbc.Container([

    # Header
    html.Div(
        html.Img(src='https://github.com/dlacksthd94/KPMG-ideathon/blob/main/app/header.png?raw=true',
                style={'height':'40%', 'width':'40%', 'margin':30})
    ),
    

    html.Div([
        # Col 1 (연관 기사)
        html.Div(
            style={'width':'28%', 'height':'120%','float':'left', 'margin':0, 'overflow':'scroll', "maxHeight": "420px"},

            children=[
                # Description
                # html.P('연관 기사 목록'),
                # html.P("기사 본문을 보려면 클릭하세요."),

                # Keyword input 

                # html.Div(
                #     children=[
                #         dbc.Input(
                #             placeholder="키워드를 입력하세요.", 
                #             type="text",
                #             size='md',
                #             style={'size':'105%'},
                #             id='keyword',
                #             debounce=True),
                #         # html.P("Hi", id='output')
                #         # dbc.FormText("키워드를 입력하세요."),
                #     ],
                #     style = {'margin':5, 'size':'105%'},
                # ),

                # News list 
                dbc.Accordion(
                    children = [
                        dbc.AccordionItem(
                                        children = [ 
                                            html.P(
                                                    f"기사 작성일: {date}",
                                                    style={'margin':3}
                                            ), 
                                            html.P(
                                                    f"기사 내용:",
                                                    style={'margin':3}
                                            ),
                                            html.P(
                                                    f" ".join(content.split('. ')[0:2]) + "...", 
                                                    style={'font-size':'90%', 'margin':1}
                                            ),
                                            html.Div(
                                                dbc.Button(
                                                    "기사 원문 보기", color="black", size='sm', href=f'{url}',
                                                ),
                                                style={'textAlign':'right'}
                                            ),
                                        ],
                                        title=f"{title}",
                                        item_id=f"item-{i+1}",
                                        style = {'border-color':'white'}
                        )
                        for i, (title, content, date, url) in enumerate(zip(titles, contents, dates, urls))
                    ],
                    start_collapsed=True,
                    id='accordion',
                    active_item=None,
                    style = {'width':'100%', 'align':'center'},
                ),
                html.Div(id="accordion-contents", className="mt-3"),
            ],
        ),


        # Col 2 (지식 그래프)
        html.Div(
            style={'width':'39%', 'height':'150%','float':'left', 'margin':10},
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
                        html.Div(children=[
                                html.H6("1. 사업 개요"),
                                html.Div(children=txt, 
                                        style={'font-size': '70%'})
                                ]),
                        html.Br(),  
                        html.Div(children=[
                                html.H6("2. 제품 및 서비스 관련 키워드"),
                                html.Div(children=[html.Span(x[0] + (", " if idx < len(ner_1)-1 else "")) for idx, x in enumerate(ner_1)], 
                                        style={'font-size': '70%'})
                                ]),
                        html.Br(),    
                        html.Div(children=[
                                html.H6("3. 원재료 및 생산설비 관련 키워드"),
                                html.Div(children=[html.Span(x[0] + (", " if idx < len(ner_2)-1 else "")) for idx, x in enumerate(ner_2)], 
                                        style={'font-size': '70%'})
                                ]),
                    ],
                    style={'width':'100%', 'height':'80%'}
                ), 

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
                                dbc.Spinner(children=[
                                    html.Iframe(id="iframe", src="about:blank", style={"width": "100%", "height": "calc(100vh - 50px)"})
                                ], type="grow"),
                            ],
                        )
                    ],
                    style={'textAlign':'right', 'font-size': '90%'}
                ),   
            ]
        )



    ])
#    
], fluid=True)


@app.callback(
    Output("accordion-contents", "children"),
    [Input("accordion", "active_item")],
)
def change_item(item):
    return 
    # return f"Item selected: {item}"

@app.callback(
    Output('graph1', 'children'),
    Input('graph1', 'selectedData'))
def display_selected_data(selectedData):
    print(json.dumps(selectedData, indent=2))
    return json.dumps(selectedData, indent=2)

@app.callback(
    Output('company_name', 'children'),
    Input('graph1', 'selectedData'))
def display_selected_data(selectedData):
    return f'{selectedData["points"][0]["text"]}'

@app.callback(
    Output("accordion", "children"), 
    [Input("keyword", "value")])
def output_text(value):
    if not value:
        return [dbc.AccordionItem(title="검색된 기사가 없습니다.")] 
    
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



if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=False, host='localhost', port=8050)