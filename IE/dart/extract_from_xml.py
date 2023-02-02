import pandas as pd
import numpy as np
import re, html
from bs4 import BeautifulSoup as BS, NavigableString, SoupStrainer
from html_table_parser import parser_functions

parser_d0350 = SoupStrainer("section-2")
section2_pattern = re.compile(r"<SECTION-2((?!<SECTION-2)[\S\s\n])*?(D-0-3-5-0)[\S\s\n]*?</SECTION-2>")

path = '../../data/kic/20210610000250.xml'
with open(path, 'r', encoding="cp949") as data_xml:
    dsd_xml = data_xml.read()
    dsd_xml = dsd_xml.replace('&cr;', '&#13;')
    dsd_xml = re.sub('(\n|\r)?</*SPAN.*?>(\n|\r)?', '', dsd_xml)
    dsd_xml = html.unescape(dsd_xml)
    section2_section = section2_pattern.search(dsd_xml)
    section2_section = section2_section.group()
    
remark_page=BS(section2_section, 'lxml', parse_only=parser_d0350).find("section-2")
remark_page



# 표 확인
table1 = remark_page.find('table', border=1)
pd.DataFrame(parser_functions.make2d(table1))

# 주석 페이지 내 표 개수 확인
len(remark_page.find_all('table', border=1 ,recursive=False))

# 페이지 내 모든 표 추출
[pd.DataFrame(parser_functions.make2d(table)) for table in remark_page.find_all('table', border=1,recursive=False)]

# 텍스트 추출하기
remark_page.find().text
list(remark_page.find().stripped_strings)

# 표를 제외한 모든 텍스트 추출하기
[list(text.stripped_strings) for text in remark_page.find_all(recursive=False) if list(text.stripped_strings) != [] and 'border="1"' not in text.prettify()]