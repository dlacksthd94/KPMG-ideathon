import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import numpy as np
import time
import datetime
from tqdm import tqdm
import os
from newspaper import Article
import re
import pororo
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
PATH_ROOT = '/home/cslim/KPMG/data/'

user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('user-agent={0}'.format(user_agent))
path='/home/cslim/selenium/chromedriver'
driver = webdriver.Chrome(path,options=options)

### get url
df_news_url = pd.DataFrame(columns=['date', 'cluster', 'url'])
list_date = [(datetime.date(2022, 12, 31) - datetime.timedelta(days=10) * i).strftime('%m.%d') for i in range(0, 13)][::-1]
idx = 0
cluster = 0
for ds, de in tqdm(zip(list_date[:-1], list_date[1:]), total=len(list_date[:-1])):
    driver.get(url=f'https://search.naver.com/search.naver?where=news&sm=tab_pge&query=폐배터리&sort=0&photo=0&field=0&pd=3&ds=2022.{ds}&de=2022.{de}&cluster_rank=69&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from20220101to20221231,a:all&start=1')
    time.sleep(1)
    html_news_group = driver.find_element(By.CSS_SELECTOR, '#main_pack > section.sc_new.sp_nnews._prs_nws > div > div.group_news > ul')#.get_attribute('innerHTML')
    list_html_news_box = html_news_group.find_elements(By.CLASS_NAME, 'bx')
    for html_news_box in list_html_news_box:
        news_title = html_news_box.find_element(By.CSS_SELECTOR, 'div.news_wrap.api_ani_send > div > a')
        news_title_date = html_news_box.find_elements(By.CSS_SELECTOR, 'span.info')[-1].text
        news_title_url = news_title.get_attribute('href')
        df_news_url.loc[idx, ['date', 'cluster', 'url']] = [news_title_date, cluster, news_title_url]
        idx += 1
        try:
            news_subtitle_cluster = html_news_box.find_element(By.CLASS_NAME, 'news_cluster')
            list_news_subtitle = news_subtitle_cluster.find_elements(By.CSS_SELECTOR, 'div.news_cluster > ul > li > span > a')
            news_subtitle_date = news_subtitle_cluster.find_elements(By.CSS_SELECTOR, 'span.sub_txt')[-1].text
            for news_subtitle in list_news_subtitle:
                news_subtitle_url = news_subtitle.get_attribute('href')
                df_news_url.loc[idx, ['date', 'cluster', 'url']] = [news_subtitle_date, cluster, news_subtitle_url]
                idx += 1
        except:
            pass
        cluster += 1
df_news_url

driver.close()

if 'df_news_url' in locals():
    df_news_url.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_url.pickle'))

df_news_url = pd.read_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_url.pickle'))

### get title and text
for idx, url in tqdm(df_news_url['url'].items(), total=df_news_url.shape[0]):
    try:
        article = Article(url, language='ko')
        article.download()
        article.parse()
        title = article.title
        content = article.text
        df_news_url.loc[idx, ['title', 'content']] = [title, content]
    except:
        pass

df_news_content = df_news_url.copy(deep=True)
df_news_content = df_news_content.dropna(how='any').reset_index(drop=True)
df_news_content['content'] = df_news_content['content'].str.replace(re.compile(r'\s+'), ' ')

if 'df_news_content' in locals():
    df_news_content.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_content.pickle'))
df_news_content = pd.read_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_content.pickle'))

### get ner
ner = pororo.Pororo(task='ner', lang='ko')

df_news_ner = df_news_content.copy(deep=True)
df_news_ner['ner'] = None
for idx, content in tqdm(df_news_ner['content'].items(), total=df_news_ner.shape[0]):
    content = content.split('. ')
    list_ne = []
    for sent in tqdm(content, leave=False):
        list_ne_temp = ner(sent)
        # list_ne_temp = ner(sent, apply_wsd=True)
        list_ne_temp = list(filter(lambda ne_pair: True if ne_pair[1] in ['ORGANIZATION', 'TERM', 'ARTIFACT', 'PERSON', 'CIVILIZATION'] and len(ne_pair[0]) > 1 else False, list_ne_temp))
        # list_ne_temp = list(filter(lambda ne_pair: True if ne_pair[1] != 'O' and len(ne_pair[0]) > 1 else False, list_ne_temp))
        list_ne.append(list_ne_temp)
    df_news_ner.at[idx, 'ner'] = list_ne

if 'df_news_ner' in locals():
    df_news_ner.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_ner.pickle'))
df_news_ner = pd.read_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_ner.pickle'))

### get embedding
se = pororo.Pororo(task='sentence_embedding', lang='ko')

df_news_embedding = df_news_ner.copy(deep=True)
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

### calculate cosine similarity
list_embedding = np.stack(df_news_embedding['embedding'])
df_cos_sim = pd.DataFrame(cosine_similarity(list_embedding))
if 'df_cos_sim' in locals():
    df_cos_sim.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_cos_sim.pickle'))
df_cos_sim = pd.read_pickle(os.path.join(PATH_ROOT, 'news', 'df_cos_sim.pickle'))

### k-means clustering
kmeans = KMeans(n_clusters=int((280 / 2) ** 0.5), max_iter=1000, tol=1e-5, n_init=1000, random_state=1000)
result = kmeans.fit_predict(df_cos_sim)
Counter(result)
df_news_final = df_news_embedding.copy(deep=True)
df_news_final['cluster'] = result

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