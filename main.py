'''
@Author: ConghaoWong
@Date: 2019-12-20 09:38:24
@LastEditors: Conghao Wong
@LastEditTime: 2020-07-15 17:02:54
@Description: main of Erina
'''
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # 去除TF输出

import time
import tensorflow as tf
import numpy as np

from matplotlib.axes._axes import _log as matplotlib_axes_logger
from PrepareTrainData import Prepare_Train_Data
from helpmethods import dir_check
from models import (
    LSTM_FC,
    Linear,
    LSTMcell,
    SS_LSTM,
    LSTM_FC_hardATT,
)

from develop_models import (
    FC_cycle,
    SS_cycle,
    SS_LSTM_beta
)


matplotlib_axes_logger.setLevel('ERROR')        # 画图警告
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"       # kNN问题
TIME = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))


def get_parser():
    parser = argparse.ArgumentParser(description='linear')

    # basic settings
    parser.add_argument('--obs_frames', type=int, default=8)
    parser.add_argument('--pred_frames', type=int, default=12)
    parser.add_argument('--test_set', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=1)

    # training data settings
    parser.add_argument('--train_type', type=str, default='one')        
    # 'one': 使用一个数据集按照分割训练集训练
    # 'all': 使用除测试外的所有数据集训练
    parser.add_argument('--train_base', type=str, default='agent')
    parser.add_argument('--frame', type=str, default='01234567')
    parser.add_argument('--train_percent', type=float, default=0.7)     # 用于训练数据的百分比
    parser.add_argument('--step', type=int, default=4)                  # 数据集滑动窗步长
    parser.add_argument('--reverse', type=int, default=True)            # 按时间轴翻转训练数据
    parser.add_argument('--add_noise', type=int, default=False)         # 训练数据添加噪声
    parser.add_argument('--noise_on_reverse', type=int, default=False)  # 是否在反转后的数据上添加噪声
    parser.add_argument('--normalization', type=int, default=False)

    # test settings
    parser.add_argument('--test', type=int, default=True)
    parser.add_argument('--start_test_percent', type=float, default=0.7)    
    parser.add_argument('--test_step', type=int, default=3)     # 训练时每test_step个epoch，test一次
    
    # training settings
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
   
    # save/load settings
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--save_model', type=int, default=True)
    parser.add_argument('--load', type=str, default='null')
    parser.add_argument('--draw_results', type=int, default=True)
    parser.add_argument('--save_base_dir', type=str, default='./logs')
    parser.add_argument('--log_dir', type=str, default='DO_NOT_CHANGE')
    parser.add_argument('--save_per_step', type=bool, default=True)

    # Linear args
    parser.add_argument('--diff_weights', type=float, default=0.95)

    # LSTM args
    parser.add_argument('--model', type=str, default='LSTM_FC')
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--save_k_results', type=bool, default=False)

    # Social args
    parser.add_argument('--max_neighbor', type=int, default=6)
    parser.add_argument('--init_position', type=float, default=20)
    parser.add_argument('--future_interaction', type=int, default=True)
    parser.add_argument('--calculate_social', type=int, default=False)
    return parser


def gpu_config(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    args = get_parser().parse_args()
    args.frame = [int(i) for i in args.frame]
    
    if args.load == 'null':
        inputs = Prepare_Train_Data(args).train_info
    else:
        inputs = 0
        args_load = np.load(args.load+'args.npy', allow_pickle=True).item()
        args_load.load = args.load
        args = args_load
    
    log_dir_current = TIME + args.model_name + args.model + str(args.test_set)
    args.log_dir = os.path.join(dir_check(args.save_base_dir), log_dir_current)
    gpu_config(args)

    if args.model == 'LSTM-ED':
        model = LSTM_ED
    elif args.model == 'LSTM_FC':
        model = LSTM_FC
    elif args.model == 'Linear':
        model = Linear
    elif args.model == 'cycle':
        model = FC_cycle
    elif args.model == 'LSTMcell':
        model = LSTMcell
    elif args.model == 'SSLSTM':
        model = SS_LSTM_beta
    elif args.model == 'test':
        model = LSTM_FC_hardATT
    elif args.model == 'sscycle':
        model = SS_cycle

    model(train_info=inputs, args=args).run_commands()


if __name__ == "__main__":
    main()
