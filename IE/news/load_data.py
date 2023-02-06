from IE.news.models import MyFastPororo
import os
import json
from tqdm import tqdm
import pickle
import joblib as jl
import torch
PATH_ROOT = '/home/cslim/KPMG/data/news'

### load data
list_dir_name = ['NIKL_NEWSPAPER_2021_v1.0/', 'NIKLNEWSPAPER_2022_v1.0/']
list_result_year = []
for dir_name in tqdm(list_dir_name):
    PATH_DIR = os.path.join(PATH_ROOT, dir_name)
    list_fine_name = os.listdir(PATH_DIR)
    list_result_group = []
    for file_name in tqdm(list_fine_name, leave=False):
        if file_name.endswith('pdf'):
            continue
        PATH_FILE = os.path.join(PATH_DIR, file_name)
        with open(PATH_FILE, "r") as f:
            news_json = json.load(f)
        list_doc = news_json['document']
        list_result_doc = []
        for doc in tqdm(list_doc, leave=False):
            list_paragraph = doc['paragraph']
            list_sent = ' '.join([paragraph['form'] for paragraph in list_paragraph[1:]]).split('. ')
            list_sent[:-1] = list(map(lambda x: x + '.', list_sent[:-1]))
            list_result_doc.append(list_sent)
        list_result_group.append(list_result_doc)
    list_result_year.append(list_result_group)
len(list_result_year[0][0][0])

with open(os.path.join(PATH_ROOT, 'news_2020_2021_data.pickle'), 'wb') as f:
    pickle.dump(list_result_year, f)

with open(os.path.join(PATH_ROOT, 'news_2020_2021_data.pickle'), 'rb') as f:
    dataset = pickle.load(f)

### prepare model
ner = MyFastPororo()
# ner.load_model()

### tokenize
list_token_year = []
for year in tqdm(dataset):
    list_token_group = []
    for group in tqdm(year, leave=False):
        list_token_doc = []
        for doc in tqdm(group, leave=False):
            list_token_sent = []
            for sent in doc:
                tokenized_sent, token_ids = ner.tokenizer(sent)
                list_token_sent.append((tokenized_sent, token_ids))
            list_token_doc.append(list_token_sent)
        list_token_group.append(list_token_doc)
    list_token_year.append(list_token_group)

with open(os.path.join(PATH_ROOT, 'news_2020_2021_token.pickle'), 'wb') as f:
    pickle.dump(list_token_year, f)

with open(os.path.join(PATH_ROOT, 'news_2020_2021_token.pickle'), 'rb') as f:
    data_token = pickle.load(f)

### make batch
list_token_sent = []
list_token_ids = []
for year in tqdm(data_token):
    for group in tqdm(year, leave=False):
        for doc in tqdm(group, leave=False):
            for token_sent, token_ids in doc:
                list_token_sent.append(token_sent)
                token_ids_padded = torch.nn.functional.pad(token_ids, (0, 500 - len(token_ids)), value=1)
                list_token_ids.append(token_ids_padded)
            # batch_token_ids = torch.stack(list_token_ids)
            # data_loader.append((list_token_sent, batch_token_ids))

BATCH_SIZE = 128
data_loader = []
for i in range(0, len(list_token_sent), BATCH_SIZE):
    batch_token_sent = list_token_sent[i:i + BATCH_SIZE]
    batch_token_ids = torch.stack(list_token_ids[i:i + BATCH_SIZE])
    data_loader.append((batch_token_sent, batch_token_ids))

with open(os.path.join(PATH_ROOT, 'news_2020_2021_batch.pickle'), 'wb') as f:
    pickle.dump(data_loader, f)

with open(os.path.join(PATH_ROOT, 'news_2020_2021_batch.pickle'), 'rb') as f:
    data_loader = pickle.load(f)
    
len(data_loader)