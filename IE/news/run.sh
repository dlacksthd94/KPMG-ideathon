conda activate kpmg
cd IE/news

python inference.py -s 0 -e 100000 -g 0
python inference.py -s 100000 -e 200000 -g 1
python inference.py -s 200000 -e 300000 -g 2
python inference.py -s 300000 -e 400000 -g 3