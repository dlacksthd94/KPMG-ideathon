from models import MyFastPororo
import os
import json
from tqdm import tqdm
import pickle
import joblib as jl
import torch
import argparse
PATH_ROOT = '/home/cslim/KPMG/data/news'

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start_idx", default=0, type=int, required=False)
parser.add_argument("-e", "--end_idx", default=50000, type=int, required=False)
parser.add_argument("-g", "--gpu-id", default=0, type=int, required=False)
args = parser.parse_args()

### prepare model
ner = MyFastPororo()
ner.load_model()
device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
ner.model.to(device)
print('model loaded')

with open(os.path.join(PATH_ROOT, 'news_2020_2021_batch.pickle'), 'rb') as f:
    data_loader = pickle.load(f)
len(data_loader) # 193934 / 41823
print('data loaded')

### inference
list_preds = []
with torch.no_grad():
    for batch_token_sent, batch_token_ids in tqdm(data_loader[args.start_idx:args.end_idx]):
    # for batch_token_sent, batch_token_ids in tqdm(data_loader[args.start_idx:args.start_idx + 10]):
        preds = ner.inference(batch_token_ids)
        list_preds.append(preds)

with open(os.path.join(PATH_ROOT, f'news_2020_2021_preds_{args.gpu_id}.pickle'), 'wb') as f:
    pickle.dump(list_preds, f)