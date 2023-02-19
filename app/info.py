
import pickle
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output
from collections import deque


def get_info():
    info = [
            
            html.Div(
                dbc.Button(
                    "사업보고서 원문",
                    id="open-offcanvas", 
                    n_clicks=0, 
                    size='sm',
                    color='dark',
                    outline=True,
                    style={'width':'25%', 'height':'50%', 'font-size':'70%'},
                ),
                style={'textAlign':'right'}
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
        ]
    return info

def filtering_dart_graph(graph, corp_name, distance=3):
    keywords = {'주요제품1':[], '사업장':[], '관계사':[], '원재료':[], '관련이슈':[]}
    stack = deque([corp_name])

    for i in range(distance):
        while stack:
            node = stack.popleft()
            neighbors = list(graph[node])
            for n in neighbors:
                keywords[graph[node][n]['kind']].append(n)

        stack += neighbors 
    

    # return keywords['주요제품1'] + keywords['관계사'] + keywords['원재료']
    return keywords