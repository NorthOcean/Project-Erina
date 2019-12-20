import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # 去除TF输出

import time
import tensorflow as tf

from matplotlib.axes._axes import _log as matplotlib_axes_logger

from PrepareTrainData import Prepare_Train_Data
from models import *

matplotlib_axes_logger.setLevel('ERROR')        # 画图警告
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"       # kNN问题
TIME = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))


def get_parser():
    parser = argparse.ArgumentParser(description='linear')

    # basic settings
    parser.add_argument('--obs_frames', type=int, default=8)
    parser.add_argument('--pred_frames', type=int, default=12)
    parser.add_argument('--total_frames', type=int, default=20) # UNUSED
    parser.add_argument('--test_set', type=int, default=0)
    parser.add_argument('--start_test_percent', type=float, default=0.7)
    parser.add_argument('--test_step', type=int, default=3)
    parser.add_argument('--temp_data_dir', type=str, default='./TEMP/')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--train_percent', type=float, default=0.8)

    # training args (for all models)
    parser.add_argument('--step', type=int, default=4)  # 数据集滑动窗步长
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_log_step', type=int, default=50)
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--load', type=str, default='null')
    parser.add_argument('--reverse', type=bool, default=True)   # 按时间轴翻转训练数据
    parser.add_argument('--add_noise', type=int, default=0)    # 训练数据添加噪声

    # save args
    parser.add_argument('--draw_results', type=bool, default=True)
    parser.add_argument('--save_base_dir', type=str, default=os.path.join('./', TIME))
    parser.add_argument('--log_dir', type=str, default=os.path.join('./', TIME))
    parser.add_argument('--save_per_step', type=bool, default=True)

    # Linear args
    parser.add_argument('--diff_weights', type=float, default=0.95)

    # LSTM args
    parser.add_argument('--model', type=str, default='LSTM-base')
    parser.add_argument('--k', type=int, default=15)
    parser.add_argument('--save_k_results', type=bool, default=False)

    # Social args
    parser.add_argument('--max_neighbor', type=int, default=6)
    parser.add_argument('--god_position', type=float, default=20)
    parser.add_argument('--future_interaction', type=bool, default=True)
    return parser


def gpu_config(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    args = get_parser().parse_args()
    args.model_name += args.model + str(args.test_set)
    args.log_dir = args.save_base_dir + args.model_name
    gpu_config(args)

    if args.model == 'LSTM-ED':
        model = LSTM_ED
    elif args.model == 'LSTM-base':
        model = LSTM_base
    elif args.model == 'LSTM-Social':
        model = LSTM_Social
    elif args.model == 'Linear':
        model = Linear

    if args.load == 'null':
        inputs = Prepare_Train_Data(args).train_agents
    else:
        inputs = 0

    model(agents=inputs, args=args).run_commands()


if __name__ == "__main__":
    main()