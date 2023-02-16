#!/usr/bin/python
# -*- coding: <encoding name> -*-

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64


# app = Dash(__name__)
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

### 연관 기사 파트 ### 
news_titles = ['안녕하세요 룰루랄라릴릴롤롤', '안녕하세요 룰루랄라릴릴롤롤','안녕하세요 룰루랄라릴릴롤롤']
news_articles = ['안녕하세요 룰루랄라릴릴롤롤', '안녕하세요 룰루랄라릴릴롤롤','안녕하세요 룰루랄라릴릴롤롤'] 







### 그래프 파트 ### 
df = px.data.iris() # iris data 불러오기
fig = px.scatter(df, x="sepal_length", y="sepal_width", color="species") # plotly를 이용한 산점도




### 기업 정보 파트 ###



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
            style={'width':'33%', 'height':'100%','float':'left'},

            children=[
            # Description
            html.P('연관 기사 목록'),
            html.P("기사 본문을 보려면 클릭하세요."),

            # News list 
            dbc.Accordion(
                [
                    dbc.AccordionItem(f"{article}",
                                    title=f"{title}",
                                    item_id=f"item-{i+1}")
                    for i, (title, article) in enumerate(zip(news_titles, news_articles))
                    
                ],
                start_collapsed=True,
                id='accordion',
                active_item=None
            ),
            html.Div(id="accordion-contents", className="mt-3"),
            ],
            
        ),


        # Col 2 (지식 그래프)
        html.Div(
            style={'width':'33%', 'height':'100%','float':'left'},
            children=[
                html.P('지식 그래프'),
                dcc.Graph(
                    id='graph1',
                    figure=fig
                )
            ],
        ),


        # Col 3 (기업 정보 요약)
        html.Div(
            style={'width':'33%', 'height':'100%','float':'left'},
            children=[
                html.P('OO 기업 정보'),
                html.Div([
                    html.H3("주요 제품")
                ], className="sub_column_1"),
                html.Div([
                    html.H3("제품 생산 원재료")
                ], className="sub_column_2"),
                html.Div([
                    html.H3("관련 키워드")
                ], className="sub_column_3")
            ]
        )



    ])
   
])


@app.callback(
    Output("accordion-contents", "children"),
    [Input("accordion", "active_item")],
)
def change_item(item):
    return f"Item selected: {item}"






if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=False, host='127.0.0.1', port=2222)