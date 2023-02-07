# -*- coding: utf-8 -*-
"""[DART] NER (아름).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14JINSgcy2WFvKjLjI3AJlWToaIP7Nkhg
"""
'ner'

import os 
import pandas as pd
import numpy as np
import re, html
# from bs4 import BeautifulSoup as BS, NavigableString, SoupStrainer
from bs4 import BeautifulSoup
from html_table_parser import parser_functions
from pororo import Pororo
import pickle
from tqdm import tqdm

"""## XML data preprocessing"""
data_path = '/home/kic/data/kic'
data_list = os.listdir(data_path)

print(data_list[0])

# 섹션별로 내용 분리하는 함수

def extract_section(path, encoding, section):
  document = []
  chapter = []
  flag = False
  section_template = f'AASSOCNOTE="\w-0-{section}-\w-\w+"'

  with open(path,'r', encoding=f'{encoding}') as fp:

    while True:
      line = fp.readline()
      
      if not line:
        break

      if re.search(section_template, line):
        document.append('\n'.join(chapter))
        chapter = []

      chapter.append(line)

  return document

ner = Pororo(task="ner", lang="ko")
srl = Pororo(task="srl", lang="ko")
pos = Pororo(task="pos", lang="ko")

sec2_list = dict() # {rcept_no: words}

for doc in tqdm(data_list):
  rcept_no = doc.replace('.xml', '')
  contents = [] # [[ner], [srl], [pos]]

  # Section 2 추출 
  try:
    sec2 = extract_section(os.path.join(data_path, doc), 'utf-8', 2)
  except:
    sec2 = extract_section(os.path.join(data_path, doc), 'cp949', 2)

  # Beautifulsoup parsing 
  soup_list = [BeautifulSoup(sec2_temp, 'html.parser') for sec2_temp in sec2]

  # 문장 추출 
  processed_sentences = []
  for soup in soup_list:
    texts = [] 
    texts += soup.find_all('p') # p tag 추출
    texts += soup.find_all('span') # span tag 추출 

    for tag in texts:
      text = tag.text.replace('\n', '').replace('&cr', '').strip()
      if text:
        if len(text) > 500:
          text = text.split('. ')
          processed_sentences += text
        else:
          processed_sentences.append(text) 

  # 뽀로로
  ner_result, srl_result, pos_result = [], [], []

  try:
    result_temp = [ner(text) for text in processed_sentences if len(text) < 500]
    ner_result += [(x,y) for ner_sentence in result_temp if (ner_sentence != 'There is NO predicate to be labeled') for x, y in ner_sentence if y not in ['O', 'QUANTITY']]
    
    result_temp = [srl(text) for text in processed_sentences if len(text) < 500]
    srl_result += [word for srl_sentence in result_temp if (srl_sentence != 'There is NO predicate to be labeled') for word in srl_sentence[0] if word[1] != '-']

    result_temp = [pos(text) for text in processed_sentences if len(text) < 500]
    pos_result += [word for pos_sentence in result_temp if (pos_sentence != 'There is NO predicate to be labeled') for word in pos_sentence if word[1] == 'NNG']
  except:
    pass
  

  sec2_list[rcept_no] = [ner_result, srl_result, pos_result]
  # del sec2S

print(f'# of processed documents: {len(sec2_list)}')

with open('dart_entities.pickle', 'wb') as f:
    pickle.dump(sec2_list, f)


