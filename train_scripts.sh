###
 # @Author: Conghao Wong
 # @Date: 1970-01-01 08:00:00
 # @LastEditors: Conghao Wong
 # @LastEditTime: 2020-08-25 19:35:34
 # @Description: Training commands
### 


cd ~/Project-Erina && 
python main.py --gpu 2 --train_type all --model SSLSTMmap --reverse 1 --rotate 5 --batch_size 5000 --epochs 300 --model_name 300MapReverseRotate5 --train_percent 0.0 --test_set 0 && 
python main.py --gpu 2 --train_type all --model SSLSTMmap --reverse 1 --rotate 5 --batch_size 5000 --epochs 300 --model_name 300MapReverseRotate5 --train_percent 0.0 --test_set 1 && 
python main.py --gpu 2 --train_type all --model SSLSTMmap --reverse 1 --rotate 5 --batch_size 5000 --epochs 300 --model_name 300MapReverseRotate5 --train_percent 0.0 --test_set 2 && 
python main.py --gpu 2 --train_type all --model SSLSTMmap --reverse 1 --rotate 5 --batch_size 5000 --epochs 300 --model_name 300MapReverseRotate5 --train_percent 0.0 --test_set 3 && 
python main.py --gpu 2 --train_type all --model SSLSTMmap --reverse 1 --rotate 5 --batch_size 5000 --epochs 300 --model_name 300MapReverseRotate5 --train_percent 0.0 --test_set 4 && 
python get_all_result.py --model_name 300MapReverseRotate5