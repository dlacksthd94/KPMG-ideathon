import pandas as pd
import numpy as np
import re, html
from bs4 import BeautifulSoup as BS, NavigableString, SoupStrainer
from html_table_parser import parser_functions
import os
from tqdm import tqdm
import pickle
from models import MyFastPororo
PATH_ROOT = '/home/cslim/KPMG/data/'

DICT_SECTION_CODE = {
    '재무제표': 'D-0-3-5-0',
    '개요': 'L-0-2-1-L1',
    '제품': 'L-0-2-2-L1',
    '원자재': 'L-0-2-3-L1',
    '매출': 'L-0-2-4-L1',
}

### select EV-related report
if os.path.exists(os.path.join(PATH_ROOT, 'kic', 'dict_corp_code.pickle')):
    with open(os.path.join(PATH_ROOT, 'kic', 'dict_corp_code.pickle'), 'rb') as f:
        dict_corp_code = pickle.load(f)
else:
    folder_path = os.path.join(PATH_ROOT, 'kic')
    list_file_name = os.listdir(folder_path)
    dict_corp_code = {}
    for file_name in tqdm(list_file_name):
        file_path = os.path.join(PATH_ROOT, 'kic', file_name)
        encoding = 'cp949' if file_name.strip('.xml') < '20220802000152' else 'utf-8'
        if file_name.endswith('xml'):
            with open(file_path, 'r', encoding=encoding, errors='ignore') as data_xml:
                dsd_xml = data_xml.read()
                xml_all = BS(dsd_xml, 'lxml', parse_only=SoupStrainer("company-name"))
                corp_code = xml_all.find('company-name').attrs['aregcik']
                dict_corp_code[corp_code] = file_name
    with open(os.path.join(PATH_ROOT, 'kic', 'dict_corp_code.pickle'), 'wb') as f:
        pickle.dump(dict_corp_code, f, protocol=4)

file_name = 'EV_processed.csv'
file_path = os.path.join(PATH_ROOT, 'dart', file_name)
df_ev = pd.read_csv(file_path)
df_ev['corp_code'] = df_ev['corp_code'].astype(str).str.pad(8, 'left', '0')
list_corp_code_ev = df_ev['corp_code'] # list of EV-related corp code

dict_xml_ev = {}
for corp_code, file_name in dict_corp_code.items():
    if corp_code in list_corp_code_ev.tolist():
        dict_xml_ev[corp_code] = file_name

### extract sector
dict_sector_product = {}
corp_code = '00612294'
file_name = dict_corp_code[corp_code]
for corp_code, file_name in tqdm(list(dict_xml_ev.items())):
    
    parser = SoupStrainer("section-2")
    section2_pattern = re.compile(rf"<SECTION-2((?!<SECTION-2)[\S\s\n])*?({DICT_SECTION_CODE['제품']})[\S\s\n]*?</SECTION-2>")

    file_path = os.path.join(PATH_ROOT, 'kic', file_name)
    encoding = 'cp949' if file_name.strip('.xml') < '20220802000152' else 'utf-8'
    with open(file_path, 'r', encoding=encoding, errors='ignore') as data_xml:
        dsd_xml = data_xml.read()
        dsd_xml = dsd_xml.replace('&cr;', '&#13;')
        dsd_xml = re.sub('(\n|\r)?</*SPAN.*?>(\n|\r)?', '', dsd_xml)
        dsd_xml = html.unescape(dsd_xml)
        section2_section = section2_pattern.search(dsd_xml)
        section2_section = section2_section.group()

    remark_page=BS(section2_section, 'lxml', parse_only=parser).find("section-2")
    # remark_page
    
    list_col_name_sector = ['사업부문', '사업구분', '분야', '품목', '대분류', '구분']
    list_df = [pd.DataFrame(parser_functions.make2d(table)) for table in remark_page.find_all('table', border=1,recursive=False)]
    
    list_col_name_product = ['주요제품', '제품', '제품명', '품목', '용도', '품목(용도)', '구분'] # if a column name contains any of these, pick it.
    # 해당 표에서 column name을 찾았으면 제일 왼쪽이 아닌 제일 오른쪽을 골라야 함.? 어떻게 할까
    # 제품 column이 사업부문 column 보다 항상 오른쪽에 있다는 게 보장되어야 함.
    # 일단 사업부문을 추출한 표에서 추출하는 게 맞고, 만약 없으면 새로 처음부터 제품만 탐색해야 함.
    
    def find_sector(list_col_name_sector, list_col_name_product, list_df):
        df_selected = None
        for col_name_sector in list_col_name_sector:
            for df in list_df:
                header = df.iloc[0].str.replace(re.compile('\s'), '').to_list()
                if col_name_sector in header:
                    df_selected = df.copy(deep=True)
                    idx_sector = header.index(col_name_sector) if col_name_sector in header else None
                    df_selected_sector = df_selected[idx_sector].str.replace(re.compile('\s+'), '')
                    bool_drop = df_selected_sector.isin(['계', ''] + list_col_name_sector) # if the word matches with the given stopwords
                    bool_drop += df_selected_sector.apply(lambda x: True if sum([stop_word in x for stop_word in ['연결조정', '차감', '내부거래', '합계', '총계', '지주회사', '수수료', '연결기준']]) else False) # if the word contains any of the given stopwords
                    bool_drop += df_selected_sector.apply(lambda x: x.replace(',', '').isdigit())
                    df_drop = df_selected[~bool_drop]
                    df_drop.iloc[:, idx_sector] = df_drop.iloc[:, idx_sector].str.replace(re.compile('(사업부?(문|분)?|부문)$'), '').str.strip()
                    df_drop.iloc[:, idx_sector] = df_drop.iloc[:, idx_sector].str.replace(r'\r', ' ').str.replace(re.compile('\s+'), ' ').str.replace(r' ?등\*?$', '')
                    for col_name_product in list_col_name_product:
                        if col_name_product in header[idx_sector + 1:]:
                            idx_product = header[idx_sector + 1:].index(col_name_product) + idx_sector + 1
                            df_selected_product = df_drop[idx_product].str.replace(re.compile('\s+'), '')
                            bool_drop = df_selected_product.isin(['계', '-', ''] + list_col_name_sector) # if the word matches with the given stopwords
                            bool_drop += df_selected_product.apply(lambda x: True if sum([stop_word in x for stop_word in ['환불', '합계', '총계', '내부거래', '영업이익', '매출', '소계']]) else False) # if the word contains any of the given stopwords
                            bool_drop += df_selected_product.apply(lambda x: x.replace(',', '').isdigit())
                            df_drop = df_drop[~bool_drop]
                            df_drop.iloc[:, idx_product] = df_drop.iloc[:, idx_product].str.replace(r'\s+', ' ').str.replace(r' ?등\*?$', '')
                            df_result = df_drop[[idx_sector, idx_product]]
                            dict_result = {}
                            for i, row in df_result.iterrows():
                                list_product = row[idx_product].split(', ')
                                list_product = list(filter(lambda x: len(x) <= 20, list_product))
                                if row[idx_sector] in dict_result:
                                    dict_result[row[idx_sector]].extend(list_product)
                                else:
                                    dict_result[row[idx_sector]] = list_product
                            for sector in dict_result:
                                dict_result[sector] = list(set(dict_result[sector]))
                            # if len(dict_result) == 1 and list(dict_result.keys())[0] == '-':
                            #     pass
                            # else:
                            return dict_result
                    df_result = df_drop[idx_sector]
                    dict_result = {}
                    for row in df_result:
                        dict_result[row] = []
                    return dict_result
        return {}
    
    dict_result_temp = find_sector(list_col_name_sector, list_col_name_product, list_df)
    dict_sector_product[corp_code] = dict_result_temp
dict_sector_product

for corp_code in dict_sector_product:
    corp_code
    for items in dict_sector_product[corp_code].items():
        items
    print('---------------------------------')

with open(os.path.join(PATH_ROOT, 'dart', 'dict_sector_product.pickle'), 'wb') as f:
    pickle.dump(dict_sector_product, f, protocol=4)

with open(os.path.join(PATH_ROOT, 'dart', 'dict_sector_product.pickle'), 'rb') as f:
    dict_sector_product = pickle.load(f)