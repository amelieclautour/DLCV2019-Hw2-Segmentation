

# download model

wget https://www.dropbox.com/s/ujxokseld80njq0/model_bestbase067.pth.tar?dl=0 -O './baseline_model' 

RESUME='./baseline_model' 

python3 test.py --resume './baseline_model' --data_dir $1 --model 'Net'

python3 save_preds.py --resume './baseline_model' --data_dir $1 --save_dir $2 --model 'Net'