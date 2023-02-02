# initiate conda
source ~/anaconda3/etc/profile.d/conda.sh

# remove env if exists
conda deactivate
conda env list

conda env remove -n kpmg
conda env list

# create env
conda create -n kpmg python=3.7 -y
conda env list

# activate env
conda activate kpmg
conda env list

# install packages
pip install -r requirements.txt 