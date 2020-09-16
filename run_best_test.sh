###
 # @Author: Conghao Wong
 # @Date: 2020-09-04 15:29:24
 # @LastEditors: Conghao Wong
 # @LastEditTime: 2020-09-16 09:30:10
 # @Description: file content
### 
python main.py --gpu 2 --load ./logs/20200905-202720NEW_500MRR3bgm0/NEW_500MRR3 --draw_results 0 --sr_enable 0 && # 0
python main.py --gpu 2 --load ./logs/20200905-202722NEW_500MRR3bgm1/NEW_500MRR3 --draw_results 0 --sr_enable 0 && # 1
python main.py --gpu 2 --load ./logs/20200905-202724NEW_500MRR3bgm2/NEW_500MRR3 --draw_results 0 --sr_enable 0 && # 2
python main.py --gpu 2 --load ./logs/20200905-202726NEW_500MRR3bgm3/NEW_500MRR3 --draw_results 0 --sr_enable 0 && # 3
python main.py --gpu 2 --load ./logs/20200905-202727NEW_500MRR3bgm4/NEW_500MRR3 --draw_results 0 --sr_enable 0 # 4

# python main.py --load ./logs/20200915-203428TOY777bgm2/TOY777 --gpu 2  # toy exp