from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pororo
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
PATH_ROOT = '/home/cslim/KPMG/data/'

### get embedding
se = pororo.Pororo(task='sentence_embedding', lang='ko')

df_news_embedding = df_news_content.copy(deep=True)
df_news_embedding['embedding'] = None
for idx, content in tqdm(df_news_embedding['content'].items(), total=df_news_embedding.shape[0]):
    content = content.split('. ')
    list_embedding = []
    for sent in tqdm(content, leave=False):
        list_embedding.append(se(sent))
    embedding_mean = np.array(list_embedding).mean(axis=0)
    df_news_embedding.at[idx, 'embedding'] = embedding_mean

if 'df_news_embedding' in locals():
    df_news_embedding.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_embedding.pickle'))
df_news_embedding = pd.read_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_embedding.pickle'))

### calculate cosine similarity and 
list_embedding = np.stack(df_news_embedding['embedding'])
df_cos_sim = pd.DataFrame(cosine_similarity(list_embedding))

if 'df_cos_sim' in locals():
    df_cos_sim.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_cos_sim.pickle'))
df_cos_sim = pd.read_pickle(os.path.join(PATH_ROOT, 'news', 'df_cos_sim.pickle'))

### k-means clustering
# list_inertia = []
# for k in tqdm(range(1, 51)):
#     kmeans = KMeans(n_clusters=k, max_iter=1000, tol=1e-5, n_init=10, random_state=1000)
#     _ = kmeans.fit_predict(np.stack(df_news_embedding['embedding']))
#     inertia = kmeans.inertia_
#     list_inertia.append(inertia)
# plt.plot(list_inertia)
# plt.savefig('/home/cslim/KPMG/clustering_inertia.png')
# plt.clf()

list_embedding_l2norm = preprocessing.normalize(np.stack(df_news_embedding['embedding']))
kmeans1 = KMeans(n_clusters=int((df_news_embedding.shape[0] / 2) ** 0.5), max_iter=1000, tol=1e-5, n_init=1000, random_state=1000)
result1 = kmeans1.fit_predict(list_embedding_l2norm)
kmeans2 = KMeans(n_clusters=int((df_news_embedding.shape[0] / 1) ** 0.5), max_iter=1000, tol=1e-5, n_init=1000, random_state=1000)
result2 = kmeans2.fit_predict(list_embedding_l2norm)
kmeans3 = KMeans(n_clusters=int((df_news_embedding.shape[0] / 0.5) ** 0.5), max_iter=1000, tol=1e-5, n_init=1000, random_state=1000)
result3 = kmeans3.fit_predict(list_embedding_l2norm)
Counter(result1)
Counter(result2)
Counter(result3)
df_news_cluster = df_news_embedding.copy(deep=True)
df_news_cluster['cluster1'] = result1
df_news_cluster['cluster2'] = result2
df_news_cluster['cluster3'] = result3
# [print(i, '\n', df_news_cluster['content'][df_news_cluster['cluster1'] == i], '\n------------') for i in range(int((df_news_embedding.shape[0] / 2) ** 0.5))]
# [print(i, '\n', df_news_cluster['content'][df_news_cluster['cluster2'] == i], '\n------------') for i in range(int((df_news_embedding.shape[0] / 1) ** 0.5))]
# [print(i, '\n', df_news_cluster['content'][df_news_cluster['cluster3'] == i], '\n------------') for i in range(int((df_news_embedding.shape[0] / 0.5) ** 0.5))]

if 'df_news_cluster' in locals():
    df_news_cluster.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_cluster.pickle'))
df_news_cluster = pd.read_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_cluster.pickle'))

### get ner
ner = pororo.Pororo(task='ner', lang='ko')

df_news_final = df_news_cluster.copy(deep=True)
df_news_final['ner'] = None
for idx, content in tqdm(df_news_final['content'].items(), total=df_news_final.shape[0]):
    content = content.split('. ')
    list_ne = []
    for sent in tqdm(content, leave=False):
        list_ne_temp = ner(sent)
        # list_ne_temp = ner(sent, apply_wsd=True)
        list_ne_temp = list(filter(lambda ne_pair: True if ne_pair[1] in ['ORGANIZATION', 'TERM', 'ARTIFACT', 'PERSON', 'CIVILIZATION'] and len(ne_pair[0]) > 1 else False, list_ne_temp))
        # list_ne_temp = list(filter(lambda ne_pair: True if ne_pair[1] != 'O' and len(ne_pair[0]) > 1 else False, list_ne_temp))
        list_ne.append(list_ne_temp)
    df_news_final.at[idx, 'ner'] = list_ne

if 'df_news_final' in locals():
    df_news_final.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_final.pickle'))
df_news_final = pd.read_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_final.pickle'))

# ### EDA
# x = 0
# for i in df_news_embedding['cluster'].unique():
#     list_idx = df_news_embedding[df_news_embedding['cluster'] == i].index
#     x += np.triu(df_cos_sim.loc[list_idx, list_idx], k=1).mean() * len(list_idx)
# x / df_news_embedding.shape[0]

# list_cos_sim = list(filter(bool, chain.from_iterable(np.triu(df_cos_sim, k=1)))) # mean 0.64
# np.array(list_cos_sim).mean()
# plt.hist(list_cos_sim, bins=20)
# plt.savefig(os.path.join('/home/cslim/KPMG/hist_news_embedding.png'))
# plt.clf()

# ### doc2vec
# from konlpy.tag import Mecab
# from gensim.models.doc2vec import TaggedDocument
# from gensim.models import doc2vec
# mecab = Mecab()

# list_content_tagged = []
# for news_id, content in tqdm(df_news_final['content'].items(), total=df_news_final.shape[0]):
#     list_content_tagged.append(TaggedDocument(tags=[news_id], words=mecab.morphs(content)))
# len(list_content_tagged)

# dim_emb = 768
# model = doc2vec.Doc2Vec(vector_size=dim_emb, alpha=0.01, min_alpha=0.001, workers=120, window=10)

# # Vocabulary 빌드
# model.build_vocab(list_content_tagged)

# # Doc2Vec 학습
# model.train(list_content_tagged, total_examples=model.corpus_count, epochs=50)

# df_cos_sim = pd.DataFrame(cosine_similarity(np.array([model.dv[i] for i in range(280)])))
# kmeans.fit_predict(np.stack(np.array([model.dv[i] for i in range(280)])))

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

# euclidean_distances(model.dv[dict_code_news_id[corp_code1][0]].reshape(1, -1), model.dv[dict_code_news_id[corp_code1][5]].reshape(1, -1))
# euclidean_distances(model.dv[dict_code_news_id[corp_code1][0]].reshape(1, -1), model.dv[dict_code_news_id[corp_code2][5]].reshape(1, -1))

# # get embedding
# list_embedding = []
# for news_id, doc in tqdm(dict_doc.items()):
#     embedding = model.infer_vector([doc])
#     list_embedding.append(embedding)
# np.array(list_embedding)