import streamlit as st
import spacy
from PIL import Image

st.set_page_config(layout="wide")

sb = st.sidebar
sb.header('검색창 👇')
text_input = sb.text_input("", placeholder='키워드를 입력해주세요', label_visibility="collapsed")
sb.write('')
sb.header('키워드 종류 필터링 👇')
options = sb.multiselect(
    '',
    ['기업', '인물', '상품', '사건'],
    ['기업', '인물', '상품', '사건'],
    label_visibility="collapsed"
)
sb.write('')
sb.header('키워드 간 관련도 가중치 👇')
edge_weight = sb.slider('키워드 간 관련도 가중치', 0, 10, 5, label_visibility="collapsed")

col1, col2 = st.columns(2)

with col1:
    #KG
    st.subheader('지식그래프')
    st.image('https://4.bp.blogspot.com/-u8zb_jrGRCI/VtyLtAhKIwI/AAAAAAAACfI/jPdE17vMbXU/s1600/ccukchat1.png')

with col2:
    #details
    st.subheader('뉴스')
    # with st.expander("See explanation"):
    st.image("https://miro.medium.com/max/1400/0*zDbB-LV-Dlm_F_PX")
    st.write('')
    st.subheader('문서 분석 결과')
    st.write('상품: 테슬라')
    st.write('사건: 전기차 폭발 사고')
    st.write('영향: 악재')
    st.write('투자 위험도: 다소 높음')