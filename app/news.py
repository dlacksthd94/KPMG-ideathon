
import pickle
import pandas as pd

with open('/home/kic/data/df_news_final.pickle', 'rb') as f:
    df = pickle.load(f)

with open('/home/kic/data/df_cos_sim.pickle', 'rb') as f:
    sim = pickle.load(f)

def get_articles(remove_redundancy=True):
    # news_titles = ['LG화학, 북미산 리튬 20만t 확보…전기차 50만대 분', '안녕하세요 룰루랄라릴릴롤롤','안녕하세요 룰루랄라릴릴롤롤']
    # news_articles = ["""
    # LG화학이 국내 전지 소재 업체 중 처음으로 북미산 리튬정광을 확보했다.\n
    # LG화학은 미국 광산 업체 피드몬트 리튬과 총 20만t 규모의 리튬 정광 구매 계약(Offtake)을 체결했다고 17일 밝혔다. 피드몬트 리튬은 캐나다 광산에서 나오는 리튬정광을 올해 3분기부터 연간 5만t씩 4년간 LG화학에 공급할 계획이다. 이는 리튬 약 3만t을 추출할 수 있는 양으로 고성능 전기차 약 50만대에 들어가는 규모다. \n
    # 피드몬트 리튬이 지분투자를 통해 간접 보유한 캐나다 퀘벡 NAL 광은, 올해 북미에서 유일하게 상업 생산이 가능한 리튬 광산이다. LG화학은 북미에서 채굴한 리튬을 북미 주요 고객에 공급하는 양극재 생산에 사용할 예정이다. \n
    # 국내 전지 소재 업체 중 북미산 리튬을 확보한 것은 LG화학이 처음이다. 리튬정광은 리튬 광석을 가공해 농축한 고순도 광물로, 배터리 핵심 원료인 수산화 리튬을 추출할 수 있다. \n
    # 북미산 리튬 정광을 사용하면 미국 미국 정부의 인플레이션 감축법(IRA)에 따른 세제 혜택 기준을 충족시키면서 이차전지 핵심 광물의 지역 편중을 완화하는데 기여할 것으로 기대된다. \n
    # LG화학은 피드몬트 리튬과 7500만 달러(약 960억원) 규모의 지분투자 계약도 체결하며 지분 약 6%를 확보했다. LG화학은 퀘벡 광산의 리튬정광 외에도 피드몬트 리튬이 미국에서 생산하는 수산화리튬 물량 연 1만t에 대한 우선협상권을 얻는 등 원재료 공급 안정성을 한층 높이게 됐다.
    # """, 
    # '안녕하세요 룰루랄라릴릴롤롤','안녕하세요 룰루랄라릴릴롤롤'] 

    if not remove_redundancy:
        article_df = df 
    else:
        article_df = df.drop_duplicates(subset='cluster', keep='first') 

    
    return list(article_df.title), list(article_df.content), list(article_df.date), list(article_df.url)