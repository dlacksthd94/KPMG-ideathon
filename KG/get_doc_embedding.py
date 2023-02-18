import pickle
import os
import pandas as pd
import numpy as np
from itertools import chain
from tqdm import tqdm
import copy
from collections import Counter
from simcse import SimCSE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from konlpy.tag import Mecab
from gensim.models.doc2vec import TaggedDocument
from gensim.models import doc2vec
from gensim.test.utils import get_tmpfile
from importlib import reload
import matplotlib.pyplot as plt
PATH_ROOT = '/home/cslim/KPMG/data/'

with open(os.path.join(PATH_ROOT, 'news', 'dict_code_news_id.pickle'), 'rb') as f:
    dict_code_news_id = pickle.load(f)
len(dict_code_news_id)

list_news_id = list(set(chain.from_iterable(dict_code_news_id.values())))
len(list_news_id)

with open(os.path.join(PATH_ROOT, 'news', f'news_2020_2021_final.pickle'), 'rb') as f:
    dict_final = pickle.load(f)

dict_doc = {}
for news_id in list_news_id:
    list_sent = dict_final[news_id]['paragraph']
    doc = ' '.join(list_sent)
    dict_doc[news_id] = doc
len(dict_doc)

### sentence embedding
import pororo
reload(pororo)
se = pororo.Pororo(task="sentence_embedding", lang="ko")

# list_embedding_sent_mean = []
# for news_id, doc in tqdm(dict_doc.items()):
#     embedding_sent = np.stack([se(sent) for sent in dict_doc[news_id].split('. ')])

list_embedding_doc = []
for news_id, doc in tqdm(dict_doc.items()):
    embedding_sent = se(dict_doc[news_id])
    list_embedding_doc.append(embedding_sent)
list_embedding = np.array(list_embedding_doc)
list_embedding.shape

df_embedding = pd.DataFrame(columns=['corp_code', 'news_id', 'embedding_doc'])
df_embedding['news_id'] = list(dict_doc.keys())
df_embedding['embedding_doc'] = list_embedding_doc
for corp_code, list_news_id in dict_code_news_id.items():
    df_embedding.loc[df_embedding['news_id'].isin(list_news_id), 'corp_code'] = corp_code

if 'df_embedding' in locals():
    df_embedding.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_embedding.pickle'))

df_embedding = pd.read_pickle(os.path.join(PATH_ROOT, 'news', 'df_embedding.pickle'))

arr_cos_sim = cosine_similarity(np.stack(df_embedding['embedding_doc']).astype(np.float32))
arr_cos_sim.shape

df_embedding['corp_code'].unique()
df_embedding[df_embedding['corp_code'] == '00414601']['news_id']
np.triu(arr_cos_sim[df_embedding[df_embedding['corp_code'] == '00414601'].index], k=1).mean() # 0.42
np.triu(arr_cos_sim[df_embedding[df_embedding['corp_code'] == '00124197'].index], k=1).mean() # 0.43
np.triu(arr_cos_sim[df_embedding[df_embedding['corp_code'] == '00125521'].index], k=1).mean() # 0.42
np.triu(arr_cos_sim[df_embedding[df_embedding['corp_code'] == '00102858'].index], k=1).mean() # 0.40
np.triu(arr_cos_sim[df_embedding[df_embedding['corp_code'] == '00105952'].index], k=1).mean() # 0.42
np.triu(arr_cos_sim[df_embedding[df_embedding['corp_code'] == '00105961'].index], k=1).mean() # 0.44
np.triu(arr_cos_sim, k=1).mean() # 0.21

df_cos_sim = pd.DataFrame(arr_cos_sim, index=df_embedding['news_id'], columns=df_embedding['news_id'])

if 'df_cos_sim' in locals():
    df_cos_sim.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_cos_sim.pickle'))

df_cos_sim = pd.read_pickle(os.path.join(PATH_ROOT, 'news', 'df_cos_sim.pickle'))

INPUT_NEWS_ID = 'NPRW2200000006.33657'
SIM_THRESHOLD = 0.77
# sr_num_sim = (df_cos_sim >= SIM_THRESHOLD).sum(axis=0)
# plt.hist(sr_num_sim, bins=100)
# plt.savefig(os.path.join(PATH_ROOT, 'news', f'hist_sim_over_{SIM_THRESHOLD}.png'))
# plt.clf()

list_sim_news_id = df_cos_sim[df_cos_sim[INPUT_NEWS_ID] >= SIM_THRESHOLD][INPUT_NEWS_ID].index
len(list_sim_news_id)

dict_final_sample = {news_id: dict_final[news_id] for news_id in list_sim_news_id}
df_cos_sim_sample = df_cos_sim.loc[list(dict_final_sample.keys()), list(dict_final_sample.keys())].round(2)

if 'dict_final_sample' in locals():
    with open(os.path.join(PATH_ROOT, 'news', f'dict_final_sample.pickle'), 'wb') as f:
        pickle.dump(dict_final_sample, f)

if 'df_cos_sim_sample' in locals():
    df_cos_sim_sample.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_cos_sim_sample.pickle'))


list_ne = {news_id: [list(filter(lambda x: True if x[1] in ['ORGANIZATION', 'TERM', 'ARTIFACT', 'PERSON'] else False, sent_ne)) for sent_ne in dict_final[news_id]['ner']] for news_id in list_sim_news_id}
len(list_ne)

if 'list_ne' in locals():
    with open(os.path.join(PATH_ROOT, 'news', f'news_ner_sample.pickle'), 'wb') as f:
        pickle.dump(list_ne, f)

# ### doc2vec
# mecab = Mecab()

# list_doc_tagged = []
# for news_id, doc in tqdm(dict_doc.items()):
#     list_doc_tagged.append(TaggedDocument(tags=[news_id], words=mecab.morphs(doc)))
# len(list_doc_tagged)

# dim_emb = 300
# model = doc2vec.Doc2Vec(vector_size=dim_emb, alpha=0.01, min_alpha=0.001, workers=120, window=10)

# # Vocabulary 빌드
# model.build_vocab(list_doc_tagged)

# # Doc2Vec 학습
# model.train(list_doc_tagged, total_examples=model.corpus_count, epochs=50)

# # 모델 저장
# model.save(os.path.join(PATH_ROOT, 'news', f'embedding_news_ev_{dim_emb}'))

# # load model
# model = doc2vec.Doc2Vec.load(os.path.join(PATH_ROOT, 'news', f'embedding_news_ev_{dim_emb}'))

# dict_code_news_id.keys()
# corp_code1 = '00126362' # 삼성SDI
# corp_code2 = '01316254' # 효성첨단소재

# dict_code_news_id[corp_code1][0]
# dict_code_news_id[corp_code2][0]

# dict_final[news_id]['paragraph']
# cosine_similarity(model.dv[dict_code_news_id[corp_code1][0]].reshape(1, -1), model.dv[dict_code_news_id[corp_code1][10]].reshape(1, -1))
# cosine_similarity(model.dv[dict_code_news_id[corp_code1][0]].reshape(1, -1), model.dv[dict_code_news_id[corp_code2][4]].reshape(1, -1))

# # euclidean_distances(model.dv[dict_code_news_id[corp_code1][0]].reshape(1, -1), model.dv[dict_code_news_id[corp_code1][5]].reshape(1, -1))
# # euclidean_distances(model.dv[dict_code_news_id[corp_code1][0]].reshape(1, -1), model.dv[dict_code_news_id[corp_code2][5]].reshape(1, -1))

# # get embedding
# list_embedding = []
# for news_id, doc in tqdm(dict_doc.items()):
#     embedding = model.infer_vector([doc])
#     list_embedding.append(embedding)
# np.array(list_embedding)