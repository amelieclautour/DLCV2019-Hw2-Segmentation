# download model



wget https://www.dropbox.com/s/svdztxcmxpfv3xp/model_bestimp.pth.tar?dl=0 -O './improved_model' 



RESUME='./improved_model' 



python3 test.py --resume './improved_model' --data_dir $1 --model 'Net_improved'



python3 save_preds.py --resume './improved_model' --data_dir $1 --save_dir $2 --model 'Net_improved'