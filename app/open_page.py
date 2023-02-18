#!/usr/bin/python
# -*- coding: <encoding name> -*-

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
import json
import os
import webbrowser

# app = Dash(__name__)
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

### 앱 레이아웃 ### 
app.layout = dbc.Container([

    # Header
    html.Div(
        html.Img(src='https://github.com/dlacksthd94/KPMG-ideathon/blob/main/app/header.png?raw=true',
                style={'height':'60%', 'width':'60%', 'margin':"40px 40px 40px 40px"})
    ),
    
    # input box
    html.Div([
        dcc.Input(
            placeholder='뉴스 링크를 입력하세요...',
            type='text',
            value='',
            style={'width':'60%', 'margin':'40px 40px 40px 100px', 'border-radius': '10px'}
        ),
    html.Button('검색', id='search-button', style={'margin':'0px 0px 0px -10px'}),
    # dcc.Location(id='url', refresh=False),
    html.Div(id='search-result')
    ])
])

# @app.callback(Output('dummy-div', 'children'), Input('url', 'pathname'))
# def open_main_page(pathname):
#     if pathname == '/app':
#         main_page_path = os.path.join(os.getcwd(), 'main_page.py')
#         webbrowser.open(main_page_path)


if __name__ == '__main__':
    # app.run_server()
    app.run_server(debug=False, host='localhost', port=8030)