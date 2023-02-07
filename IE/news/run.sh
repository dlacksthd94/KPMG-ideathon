conda activate kpmg
cd IE/news

python inference.py -s 0 -e 50000 -g 0
python inference.py -s 50000 -e 100000 -g 1
python inference.py -s 100000 -e 150000 -g 2
python inference.py -s 150000 -e 200000 -g 3