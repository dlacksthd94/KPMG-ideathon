import pickle
import os
import pandas as pd
from itertools import chain
from tqdm import tqdm
import copy
from collections import Counter

PATH_ROOT = '/home/cslim/KPMG/data/'
with open(os.path.join(PATH_ROOT, 'html_EV.pickle'), 'rb') as f:
    dart_ner = pickle.load(f)

df_corp_ev = pd.read_csv(os.path.join(PATH_ROOT, 'dart', 'EV_processed.csv'))
df_corp_ev['corp_code'] = df_corp_ev['corp_code'].astype(str).str.pad(8, 'left', '0')
dict_code_name = dict(df_corp_ev[['corp_code', 'corp_name']].to_numpy())

with open(os.path.join(PATH_ROOT, 'news', f'news_2020_2021_final.pickle'), 'rb') as f:
    dict_final = pickle.load(f)

dict_section = {
    'summary': 0,
    'product': 1,
}

list_corp_code = list(dart_ner.keys())
dict_code_news_id = {corp_code: [] for corp_code in list_corp_code}
for news_id, doc in tqdm(dict_final.items()):
    for corp_code in list_corp_code:
        corp_name = dict_code_name[corp_code]
        list_sent_ne = doc['ner']
        target = tuple([corp_name, 'ORGANIZATION'])
        if target in list(chain.from_iterable(list_sent_ne)):
            dict_code_news_id[corp_code].append(news_id)
for corp_code in dict_code_news_id.keys():
    dict_code_name[corp_code], len(dict_code_news_id[corp_code])

if 'dict_code_news_id' in locals():
    with open(os.path.join(PATH_ROOT, 'news', 'dict_code_news_id.pickle'), 'wb') as f:
        pickle.dump(dict_code_news_id, f)

with open(os.path.join(PATH_ROOT, 'news', 'dict_code_news_id.pickle'), 'rb') as f:
    dict_code_news_id = pickle.load(f)
len(dict_code_news_id)

dict_code_news_id_ne = {}
for corp_code in tqdm(dict_code_news_id.keys()):
    corp_name = dict_code_name[corp_code]
    dict_code_news_id_ne[corp_code] = {}
    for news_id in tqdm(dict_code_news_id[corp_code], leave=False):
        list_sent_ne = dict_final[news_id]['ner']
        for sent_ne in list_sent_ne:
            target = tuple([corp_name, 'ORGANIZATION'])
            if target in sent_ne:
                list_ne_related = list(filter(lambda ne_pair: True if ne_pair[1] in ['ARTIFACT', 'TERM', 'ORGANIZATION', 'PERSON'] else False, sent_ne))
                if list_ne_related:
                    dict_code_news_id_ne[corp_code][news_id] = list_ne_related
len(dict_code_news_id_ne)

dict_code_news_id_ne_count = {}
for corp_code, dict_news_id_ne in tqdm(dict_code_news_id_ne.items()):
    corp_name = dict_code_name[corp_code]
    list_all = list(chain.from_iterable(dict_news_id_ne.values()))
    list_all = list(filter(lambda ne_pair: False if len(ne_pair[0]) <= 1 else True, list_all))
    list_product = list(filter(lambda ne_pair: True if ne_pair[1] in ['ARTIFACT', 'TERM'] else False, list_all))
    list_org = list(filter(lambda ne_pair: True if ne_pair[1] in ['ORGANIZATION'] else False, list_all))
    list_person = list(filter(lambda ne_pair: True if ne_pair[1] in ['PERSON'] else False, list_all))
    ne_product_count = Counter(list_product)
    ne_org_count = Counter(list_org)
    ne_person_count = Counter(list_person)
    dict_code_news_id_ne_count[corp_code] = {}
    dict_code_news_id_ne_count[corp_code]['product'] = ne_product_count
    dict_code_news_id_ne_count[corp_code]['org'] = ne_org_count
    dict_code_news_id_ne_count[corp_code]['person'] = ne_person_count

dict_code_news_id_ne_count['00126362']['product'].most_common(10) # 삼성SDI
dict_code_news_id_ne_count['00126362']['org'].most_common(10)
dict_code_news_id_ne_count['00126362']['person'].most_common(10)

if 'dict_code_news_id_ne_count' in locals():
    with open(os.path.join(PATH_ROOT, 'news', 'dict_code_news_id_ne_count.pickle'), 'wb') as f:
        pickle.dump(dict_code_news_id_ne_count, f)

with open(os.path.join(PATH_ROOT, 'news', 'dict_code_news_id_ne_count.pickle'), 'rb') as f:
    dict_code_news_id_ne_count = pickle.load(f)
len(dict_code_news_id_ne_count)
