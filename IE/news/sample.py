from models import MyFastPororo

# prepare model
ner = MyFastPororo()
ner.load_model()

# inference
text = '삼정 KPMG에서 주최하는 아이디어톤에 서울대 데이터사이언스대학원 (GSDS) 의 박건도, 서아름, 손성욱, 임찬송, 최유림, 허상우 학생이 참여하였습니다.'

tokenized_sent, token_ids = ner.tokenizer(text)
preds = ner.inference(token_ids)[0]
result = ner.post_process(tokenized_sent, preds)
print(result)