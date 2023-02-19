
import pickle
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import datetime

def get_articles(df, remove_redundancy=True):

    if not remove_redundancy:
        article_df = df 
    else:
        article_df = df.drop_duplicates(subset='cluster_cosine', keep='first') 

    return list(article_df.title), list(article_df.content), list(article_df.date), list(article_df.url)


def get_accodion_items(new_df):
    titles, contents, dates, urls = get_articles(new_df)
    accodion = \
    [
        dbc.Accordion(
            children=[
                dbc.AccordionItem(
                    children=[
                        html.P(
                                f" ".join(content.split('. ')[0:2]) + "...", 
                                style={'font-size':'90%', 'margin':1}
                        ),
                        html.Div(
                            dbc.Button(
                                "기사 원문 보기", color="black", size='sm', href=f'{url}', outline='dark'
                            ),
                            style={'textAlign':'right'}
                        )
                    ],
                    title=f"{title}  ({date.strftime('%y-%m-%d')})",
                    item_id=f"item-{i+1}",
                    style = {'border-color':'white'}
                )
                for i, (title, content, date, url) in enumerate(zip(titles, contents, dates, urls))  
            ],
            start_collapsed=True,
            # id='accordion',
            active_item=None,
            style = {'width':'100%', 'align':'center'},
        ),
        html.Div(id="accordion-contents", className="mt-3")
    ]
    return accodion

    