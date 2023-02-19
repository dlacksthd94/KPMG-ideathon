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
from sklearn.metrics.pairwise import cosine_similarity
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
df_news_url = pd.DataFrame(columns=['date', 'url'])
list_date = [(datetime.date(2022, 12, 31) - datetime.timedelta(days=10) * i).strftime('%m.%d') for i in range(0, 16)][::-1]
idx = 0
for ds, de in tqdm(zip(list_date[:-1], list_date[1:]), total=len(list_date[:-1])):
    driver.get(url=f'https://search.naver.com/search.naver?where=news&sm=tab_pge&query=폐배터리&sort=0&photo=0&field=0&pd=3&ds=2022.{ds}&de=2022.{de}&cluster_rank=69&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:r,p:from20220101to20221231,a:all&start=1')
    time.sleep(1)
    html_news_group = driver.find_element(By.CSS_SELECTOR, '#main_pack > section.sc_new.sp_nnews._prs_nws > div > div.group_news > ul')#.get_attribute('innerHTML')
    list_html_news_box = html_news_group.find_elements(By.CLASS_NAME, 'bx')
    for html_news_box in list_html_news_box:
        news_title = html_news_box.find_element(By.CSS_SELECTOR, 'div.news_wrap.api_ani_send > div > a')
        news_title_date = html_news_box.find_elements(By.CSS_SELECTOR, 'span.info')[-1].text
        news_title_url = news_title.get_attribute('href')
        df_news_url.loc[idx, ['date', 'url']] = [news_title_date, news_title_url]
        idx += 1
        try:
            news_subtitle_cluster = html_news_box.find_element(By.CLASS_NAME, 'news_cluster')
            list_news_subtitle = news_subtitle_cluster.find_elements(By.CSS_SELECTOR, 'div.news_cluster > ul > li > span > a')
            news_subtitle_date = news_subtitle_cluster.find_elements(By.CSS_SELECTOR, 'span.sub_txt')[-1].text
            for news_subtitle in list_news_subtitle:
                news_subtitle_url = news_subtitle.get_attribute('href')
                df_news_url.loc[idx, ['date', 'url']] = [news_subtitle_date, news_subtitle_url]
                idx += 1
        except:
            pass
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
df_news_content = df_news_content[df_news_content['content'].str.contains('폐배터리')].reset_index(drop=True) # drop incorrect articles

if 'df_news_content' in locals():
    df_news_content.to_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_content.pickle'))
df_news_content = pd.read_pickle(os.path.join(PATH_ROOT, 'news', 'df_news_content.pickle'))