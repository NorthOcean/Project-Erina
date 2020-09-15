###
 # @Author: Conghao Wong
 # @Date: 1970-01-01 08:00:00
 # @LastEditors: Conghao Wong
 # @LastEditTime: 2020-09-15 09:55:54
 # @Description: Training commands
### 


cd ~/Project-Erina && 
python main.py --gpu 1 --train_type all --model bgm --reverse 1 --rotate 3 --batch_size 20000 --epochs 500 --model_name NEW_500MRR3 --train_percent 0.0 --test_set 0 && 
python main.py --gpu 1 --train_type all --model bgm --reverse 1 --rotate 3 --batch_size 20000 --epochs 500 --model_name NEW_500MRR3 --train_percent 0.0 --test_set 1 && 
python main.py --gpu 1 --train_type all --model bgm --reverse 1 --rotate 3 --batch_size 20000 --epochs 500 --model_name NEW_500MRR3 --train_percent 0.0 --test_set 2 && 
python main.py --gpu 1 --train_type all --model bgm --reverse 1 --rotate 3 --batch_size 20000 --epochs 500 --model_name NEW_500MRR3 --train_percent 0.0 --test_set 3 && 
python main.py --gpu 1 --train_type all --model bgm --reverse 1 --rotate 3 --batch_size 20000 --epochs 500 --model_name NEW_500MRR3 --train_percent 0.0 --test_set 4 # && 
# python get_all_result.py --model_name NEW_500MRR3