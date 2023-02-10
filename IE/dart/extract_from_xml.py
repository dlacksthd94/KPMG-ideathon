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
dict_sector = {}
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

    # xml_all = BS(dsd_xml, 'lxml')

    # company_name = xml_all.find('company-name').text
    # company_name
    # company_code = xml_all.find('company-name').attrs['aregcik']
    # company_code
    # doc_name = xml_all.find('document-name').text
    # doc_name
    # report_date = file_name[:8]
    # report_date
    remark_page=BS(section2_section, 'lxml', parse_only=parser).find("section-2")
    # remark_page
    
    list_col_name = ['사업부문', '사업구분', '분야', '품목', '대분류'] # 구분
    list_df = [pd.DataFrame(parser_functions.make2d(table)) for table in remark_page.find_all('table', border=1,recursive=False)]
    
    def find_sector(list_col_name, list_df):
        df_selected = None
        for col_name in list_col_name:
            for df in list_df:
                header = df.iloc[0].str.replace(re.compile('\s'), '').to_list()
                if col_name in header:
                    df_selected = df.copy(deep=True)
                    i = header.index(col_name) if col_name in header else None
                    df_selected_rpl = df_selected[i].str.replace(re.compile('\s+'), '')
                    bool_drop = df_selected_rpl.isin(['계', '-', ''] + list_col_name) # if the word matches with the given stopwords
                    bool_drop += df_selected_rpl.apply(lambda x: True if sum([stop_word in x for stop_word in ['연결조정', '차감', '내부거래', '합계', '총계', '지주회사', '수수료', '연결기준']]) else False) # if the word contains any of the given stopwords
                    df_sector = df_selected[i][~bool_drop]
                    df_sector = df_sector.str.replace(re.compile('(사업부?(문|분)?|부문)$'), '').str.strip()
                    df_sector = df_sector.str.replace(r'\r', ' ').str.replace(re.compile('\s+'), ' ')
                    list_sector = list(set(df_sector.to_list()))
                    if len(list_sector) == 1 and list_sector[0] == '-':
                        pass
                    else:
                        return list_sector
        return []
    
    list_sector_temp = find_sector(list_col_name, list_df)
    dict_sector[corp_code] = list_sector_temp
dict_sector

with open(os.path.join(PATH_ROOT, 'dart', 'dict_sector.pickle'), 'wb') as f:
    pickle.dump(dict_sector, f, protocol=4)

with open(os.path.join(PATH_ROOT, 'dart', 'dict_sector.pickle'), 'rb') as f:
    dict_sector = pickle.load(f)
