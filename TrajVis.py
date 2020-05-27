# * @Author: ConghaoWong
# * @Date: 2020-05-12 14:11:06 
# * @Last Modified by:   ConghaoWong 
# * @Last Modified time: 2020-05-12 14:11:06

import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PrepareTrainData import Prepare_Train_Data

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
    parser.add_argument('--train_percent', type=float, default=0.7)     # 用于训练数据的百分比
    parser.add_argument('--step', type=int, default=4)                  # 数据集滑动窗步长
    parser.add_argument('--reverse', type=int, default=False)            # 按时间轴翻转训练数据
    parser.add_argument('--add_noise', type=int, default=False)         # 训练数据添加噪声
    parser.add_argument('--noise_on_reverse', type=int, default=False)  # 是否在反转后的数据上添加噪声
    parser.add_argument('--normalization', type=int, default=False)


   
    # save/load settings
    parser.add_argument('--model_name', type=str, default='model')
    parser.add_argument('--save_model', type=int, default=True)
    parser.add_argument('--load', type=str, default='null')
    parser.add_argument('--draw_results', type=int, default=True)
    parser.add_argument('--save_base_dir', type=str, default='./logs')
    parser.add_argument('--log_dir', type=str, default='DO_NOT_CHANGE')
    parser.add_argument('--save_per_step', type=bool, default=True)

    # Social args
    parser.add_argument('--max_neighbor', type=int, default=6)
    parser.add_argument('--god_position', type=float, default=20)
    parser.add_argument('--future_interaction', type=int, default=True)
    parser.add_argument('--calculate_social', type=int, default=False)
    return parser


def draw_one_traj(agent_index, all_agents):
    agent = all_agents[agent_index]
    start = agent.start_frame
    end = agent.end_frame

    all_neighbor = set([])
    for nei in agent.neighbor_list[start:end]:
        all_neighbor = all_neighbor | set(nei)
    all_neighbor -= set([agent_index])

    # plt.figure()
    for frame in tqdm(range(start, end), desc='Agent{}'.format(agent_index)):
        nei_current = agent.neighbor_list[frame]

        plt.figure()
        
        for nei in all_neighbor:
            plt.plot(all_agents[nei].traj[frame][0], all_agents[nei].traj[frame][1], 'o', markersize=nei in nei_current and 16 or 8)
        
        plt.plot(agent.traj[frame][0], agent.traj[frame][1], '*', markersize=16)
        
        plt.axis('scaled')
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.title('{}-{}'.format(agent_index, frame))
        plt.savefig('./vislogs/{}-{}.png'.format(agent_index, frame))
        plt.close()


def main():
    args = get_parser().parse_args()
    all_agents = Prepare_Train_Data(args).all_agents

    for i, _ in enumerate(all_agents):
        draw_one_traj(i, all_agents)
    print('!')


if __name__ == '__main__':
    main()