from IE.news.models import MyFastPororo
import os
import json
from tqdm import tqdm
import pickle
import joblib as jl
import torch
import numpy as np
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
list_token_sent = []
list_token_ids = []
for year in tqdm(dataset):
    for group in tqdm(year, leave=False):
        for doc in tqdm(group, leave=False):
            for sent in doc:
                token_sent, token_ids = ner.tokenizer(sent)
                list_token_sent.append(token_sent)
                list_token_ids.append(token_ids)
list_token_ids = list(map(lambda x: x.numpy(), list_token_ids))

with open(os.path.join(PATH_ROOT, 'news_2020_2021_token_sent.pickle'), 'wb') as f:
    pickle.dump(list_token_sent, f)

with open(os.path.join(PATH_ROOT, 'news_2020_2021_token_sent.pickle'), 'rb') as f:
    list_token_sent = pickle.load(f)

with open(os.path.join(PATH_ROOT, 'news_2020_2021_token_ids.pickle'), 'wb') as f:
    pickle.dump(list_token_ids, f)

with open(os.path.join(PATH_ROOT, 'news_2020_2021_token_ids.pickle'), 'rb') as f:
    list_token_ids = pickle.load(f)

### check
for toke_sent, token_ids in tqdm(zip(list_token_sent, list_token_ids), total=len(list_token_ids)):
    token_ids = token_ids[1:-1]
    assert len(toke_sent.split()) == len(token_ids)

# ### padding
# MAX_LENGTH = 500
# list_token_ids = []
# list_token_sent_padded = []
# token_ids_concat = np.array([], dtype=int)
# token_sent_concat = ''
# for toke_sent, token_ids in tqdm(zip(list_token_sent, list_token_ids), total=len(list_token_ids)):
#     token_ids = token_ids[1:-1]
#     if token_ids.shape[0] >= MAX_LENGTH:
#         token_ids = token_ids[:MAX_LENGTH]
#         token_ids
#     if token_ids_concat.shape[0] + token_ids.shape[0] <= MAX_LENGTH:
#         token_ids_concat = np.concatenate([token_ids_concat, token_ids])
#         token_sent_concat += toke_sent + '<sep>'
#     else:
#         token_ids_concat = np.concatenate([[0], token_ids_concat, [2]])
#         token_ids_padded = torch.nn.functional.pad(torch.tensor(token_ids_concat), (0, MAX_LENGTH - len(token_ids_concat)), value=1)
#         list_token_ids_padded.append(token_ids_padded)
#         # token_sent_concat = token_sent_concat + '<pad> ' * (MAX_LENGTH - len(token_ids_concat))
#         token_sent_concat = token_sent_concat.strip().strip('<sep>')
#         list_token_sent_padded.append(token_sent_concat)
#         # reset token_ids_concat
#         token_ids_concat = token_ids
#         token_sent_concat = toke_sent + '<sep>'
# token_ids_concat = np.concatenate([[0], token_ids_concat, [2]])
# token_ids_padded = torch.nn.functional.pad(torch.tensor(token_ids_concat), (0, MAX_LENGTH - len(token_ids_concat)), value=1)
# list_token_ids_padded.append(token_ids_padded)
# # token_sent_concat = token_sent_concat + '<pad> ' * (MAX_LENGTH - len(token_ids_concat))
# token_sent_concat = token_sent_concat.strip().strip('<sep>')
# list_token_sent_padded.append(token_sent_concat)

# ### make batch
# BATCH_SIZE = 32
# list_batch = []
# for i in tqdm(range(0, len(list_token_sent_padded), BATCH_SIZE)):
#     batch_token_sent = list_token_sent_padded[i:i + BATCH_SIZE]
#     batch_token_ids = torch.stack(list_token_ids_padded[i:i + BATCH_SIZE])
#     list_batch.append((batch_token_sent, batch_token_ids))
# len(list_batch)

### make batch
MAX_INPUT_LENGTH = 500
BATCH_SIZE = 64
list_batch = []
for i in tqdm(range(0, len(list_token_sent), BATCH_SIZE)):
    batch_token_sent = list_token_sent[i:i + BATCH_SIZE]
    batch_token_ids = list_token_ids[i:i + BATCH_SIZE]
    max_len = max([len(token_ids) for token_ids in batch_token_ids])
    if max_len >= MAX_INPUT_LENGTH:
        for j in range(len(batch_token_sent)):
            token_sent = batch_token_sent[j]
            token_ids = batch_token_ids[j]
            if token_ids.shape[0] >= MAX_INPUT_LENGTH:
                token_sent_split = token_sent.split()[:MAX_INPUT_LENGTH]
                for k, token in enumerate(token_sent_split[::-1]):
                    if token.startswith('▁'):
                        break
                    end_index = len(token_sent_split) - k - 2
                batch_token_sent[j] = ' '.join(token_sent_split[:end_index])
                batch_token_ids[j] = np.concatenate([token_ids[:end_index + 1], [2]])
        max_len = max([len(token_ids) for token_ids in batch_token_ids])
    batch_token_ids = torch.tensor([np.pad(token_ids, (0, max_len - len(token_ids)), constant_values=1) for token_ids in batch_token_ids])
    list_batch.append((batch_token_sent, batch_token_ids))
len(list_batch)

with open(os.path.join(PATH_ROOT, 'news_2020_2021_batch.pickle'), 'wb') as f:
    pickle.dump(list_batch, f)

with open(os.path.join(PATH_ROOT, 'news_2020_2021_batch.pickle'), 'rb') as f:
    list_batch = pickle.load(f)