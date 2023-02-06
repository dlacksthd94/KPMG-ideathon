from models import MyFastPororo
import os
import json
from tqdm import tqdm
import pickle
import joblib as jl
import torch
import argparse
PATH_ROOT = '/home/cslim/KPMG/data/news'

ner = MyFastPororo()

for gpu_id in range(4):
    with open(os.path.join(PATH_ROOT, f'news_2020_2021_preds_{gpu_id}.pickle'), 'rb') as f:
        list_preds = pickle.load(f)

    ### post-process
    list_ner = []
    for news_year in tqdm(dataset):
        list_ner_year = []
        for group in tqdm(news_year, leave=False):
            list_ner_group = []
            for doc in tqdm(group, leave=False):
                list_ner_doc = []
                for para in doc:
                    try:
                        tokenized_sent, token_ids = ner.tokenizer(para)
                        preds = ner.inference(token_ids)
                        for pred in preds:
                            ner_para = ner.post_process(tokenized_sent, pred)
                        list_ner_doc.append(ner_para)
                    except:
                        ner_para = []
                list_ner_group.append(list_ner_doc)
            list_ner_year.append(list_ner_group)
        list_ner.append(list_ner_year)
    len(list_ner)
