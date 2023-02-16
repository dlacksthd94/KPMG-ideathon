from models import MyFastPororo
import os
from tqdm import tqdm
import pickle
from itertools import chain
import json
PATH_ROOT = '/home/cslim/KPMG/data/news'

ner = MyFastPororo()
ner.load_model()

with open(os.path.join(PATH_ROOT, f'news_2020_2021_data.pickle'), 'rb') as f:
    list_data = pickle.load(f)

# with open(os.path.join(PATH_ROOT, f'news_2020_2021_token_sent.pickle'), 'rb') as f:
#     list_token_sent = pickle.load(f)
# len(list_token_sent) # 24823444

with open(os.path.join(PATH_ROOT, f'news_2020_2021_token_ids.pickle'), 'rb') as f:
    list_token_ids = pickle.load(f)
    
with open(os.path.join(PATH_ROOT, f'news_2020_2021_batch.pickle'), 'rb') as f:
    list_batch = pickle.load(f)

list_result = []
for gpu_id in tqdm(range(4)):
    with open(os.path.join(PATH_ROOT, f'news_2020_2021_preds_{gpu_id}.pickle'), 'rb') as f:
        list_preds = pickle.load(f)
    
    # start_idx = gpu_id * 10000
    # for (token_sent_batch, token_ids_batch), pred_batch in tqdm(zip(list_batch[start_idx:], list_preds), total=len(list_preds)):
    #     for list_token_sent, list_pred in zip(token_sent_batch, pred_batch):
    #         list_token_sent = list_token_sent.split('<sep>')
    #         list_token_sent_len = list(map(lambda x: len(x.split()), list_token_sent))
    #         list_pred = list_pred[1:1 + sum(list_token_sent_len)]
    #         start_idx = 0
    #         for i, token_sent in enumerate(list_token_sent):
    #             token_sent_len = list_token_sent_len[i]
    #             pred = list_pred[start_idx:start_idx + token_sent_len]
    #             pred = np.concatenate([[0], pred, [0]])
    #             start_idx = token_sent_len
    #             list_result = ner.post_process(token_sent, pred)
    #             list_result
    #         # list_ne = list(filter(lambda x: True if x[1] != 'O' else False, list_result))
    
    start_idx = gpu_id * 100000
    for (token_sent_batch, token_ids_batch), pred_batch in tqdm(zip(list_batch[start_idx:], list_preds), total=len(list_preds)):
        for token_sent, pred in zip(token_sent_batch, pred_batch):
            result = ner.post_process(token_sent, pred)
            list_result.append(result)

assert len(list_result) == 24823444

with open(os.path.join(PATH_ROOT, 'news_2020_2021_result.pickle'), 'wb') as f:
    pickle.dump(list_result, f)

with open(os.path.join(PATH_ROOT, 'news_2020_2021_result.pickle'), 'rb') as f:
    list_result = pickle.load(f)

assert len(list_result) == len(list(chain.from_iterable(chain.from_iterable(chain.from_iterable(list_data)))))

i = 0
list_result_aligned = []
for year in tqdm(list_data):
    list_year = []
    for group in tqdm(year, leave=False):
        list_group = []
        for doc in tqdm(group, leave=False):
            num_sent = len(doc)
            list_sent = list_result[i:i + num_sent]
            i += num_sent
            list_group.append(list_sent)
        list_year.append(list_group)
    list_result_aligned.append(list_year)
assert i == len(list_result)

if 'list_result_aligned' in locals():
    with open(os.path.join(PATH_ROOT, 'news_2020_2021_result_aligned.pickle'), 'wb') as f:
        pickle.dump(list_result_aligned, f)

with open(os.path.join(PATH_ROOT, 'news_2020_2021_result_aligned.pickle'), 'rb') as f:    
    list_result_aligned = pickle.load(f)
len(list_result_aligned)

### align original data with ne result
list_doc_ner = list(chain.from_iterable(chain.from_iterable(list_result_aligned))) # 1707622
list_doc = list(chain.from_iterable(chain.from_iterable(list_data))) # 1707622

i = 0
dict_final = {}
list_dir_name = ['NIKL_NEWSPAPER_2021_v1.0/', 'NIKLNEWSPAPER_2022_v1.0/']
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
        list_doc_temp = news_json['document']
        for doc in tqdm(list_doc_temp, leave=False):
            doc['paragraph'] = list_doc[i]
            doc['ner'] = [list(set([ne_pair for ne_pair in ne_sent if ne_pair[1] != 'O'])) for ne_sent in list_doc_ner[i]]
            new_id = doc.pop('id')
            dict_final[new_id] = doc
            i += 1

len(dict_final) # 1707622
dict_final[list(dict_final.keys())[-1]]

if 'dict_final' in locals():
    with open(os.path.join(PATH_ROOT, f'news_2020_2021_final.pickle'), 'wb') as f:
        pickle.dump(dict_final, f)