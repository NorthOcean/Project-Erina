'''
@Author: ConghaoWong
@Date: 2019-12-20 09:39:02
LastEditors: Conghao Wong
LastEditTime: 2020-08-23 00:30:41
@Description: file content
'''
import os
import random
USE_SEED = True
SEED = 10

import numpy as np
from tqdm import tqdm

from helpmethods import dir_check, list2array, predict_linear_for_person, calculate_ADE_FDE_numpy

import matplotlib.pyplot as plt

class Prepare_Train_Data():
    def __init__(self, args, save=True):
        self.args = args
        self.obs_frames = args.obs_frames
        self.pred_frames = args.pred_frames
        self.total_frames = self.pred_frames + self.obs_frames
        self.step = args.step

        self.max_neighbor = args.max_neighbor
        self.init_position = np.array([args.init_position, args.init_position])
        self.god_past_traj = np.stack([self.init_position for _ in range(self.obs_frames)])
        self.god_future_traj = np.stack([self.init_position for _ in range(self.pred_frames)])

        self.log_dir = dir_check(args.log_dir)
        self.save_file_name = args.model_name + '_{}.npy'
        self.save_path = os.path.join(self.log_dir, self.save_file_name)
        self.train_info = self.get_train_and_test_agents()
        
    def get_train_and_test_agents(self):
        dir_check('./dataset_npz/')
        self.npy_file_base_path = './dataset_npz/{}/data.npz'

        if self.args.train_type == 'one':
            train_list = [self.args.test_set]
            test_list = [self.args.test_set]
        
        elif self.args.train_type == 'all':
            # self.args.normalization = True
            test_list = [self.args.test_set]
            train_list = [i for i in range(5) if not i == self.args.test_set]
        
        data_managers_train = []
        for dataset in train_list:
            data_managers_train.append(self.get_agents_from_dataset(dataset)) 

        data_managers_test = []
        for dataset in test_list:
            data_managers_test.append(self.get_agents_from_dataset(dataset)) 

        sample_number_original = 0
        sample_time = 1
        for dm in data_managers_train:
            sample_number_original += dm.person_number
            
        if self.args.train_type == 'one':
            index = set([i for i in range(sample_number_original)])
            if USE_SEED:
                random.seed(SEED)
            train_index = random.sample(index, int(sample_number_original * self.args.train_percent))
            test_index = list(index - set(train_index))
            
            test_agents = self.sample_data(data_managers_train[0], test_index)
            train_agents = []
            if self.args.reverse:
                train_agents += self.sample_data(data_managers_train[0], train_index, reverse=True, desc='Preparing reverse data')
                sample_time += 1

            if self.args.add_noise:                
                for repeat in tqdm(range(self.args.add_noise), desc='Preparing noise data'):
                    train_agents += self.sample_data(data_managers_train[0], train_index, add_noise=True, use_time_bar=False)
                    sample_time += 1
        
        elif self.args.train_type == 'all':
            train_agents = []
            for dm in data_managers_train:
                train_agents += self.sample_data(dm, person_index='all', random_sample=0.5)

            if self.args.reverse:
                for dm in data_managers_train:
                    train_agents += self.sample_data(dm, person_index='all', reverse=True, use_time_bar=False)
                sample_time += 1
                
            test_agents = self.sample_data(data_managers_test[0], person_index='all')
        
        train_info = dict()
        # train_info['all_agents'] = all_data
        train_info['train_data'] = train_agents
        train_info['test_data'] = test_agents
        train_info['train_number'] = len(train_agents)
        train_info['sample_time'] = sample_time  

        # # test_options
        # np.save('./test_data_seed10/train{}.npy'.format(self.args.test_set), train_data)
        # np.save('./test_data_seed10/test{}.npy'.format(self.args.test_set), test_data)
        # raise
        return train_info

    def data_loader(self, dataset_index):
        """
        从原始csv文件中读取数据
            return: person_data, frame_data
        """
        # dataset_index = self.args.test_set
        dataset_dir = [
            './data/eth/univ',
            './data/eth/hotel',
            './data/ucy/zara/zara01',
            './data/ucy/zara/zara02',
            './data/ucy/univ/students001'
        ]

        dataset_xy_order = [
            [3, 2],
            [2, 3],
            [3, 2],
            [3, 2],
            [2, 3],
        ]

        dataset_dir_current = dataset_dir[dataset_index]
        order = dataset_xy_order[dataset_index]

        csv_file_path = os.path.join(dataset_dir_current, 'true_pos_.csv')
        data = np.genfromtxt(csv_file_path, delimiter=',').T 

        # 加载数据（使用帧排序）
        frame_data = {}
        frame_list = set(data.T[0])
        for frame in frame_list:
            index_current = np.where(data.T[0] == frame)[0]
            frame_data[str(frame)] = np.column_stack([
                data[index_current, 1],
                data[index_current, order[0]],
                data[index_current, order[1]],
            ])

        # 加载数据（使用行人编号排序）
        person_data = {}
        person_list = set(data.T[1])
        for person in person_list:
            index_current = np.where(data.T[1] == person)[0]
            person_data[str(person)] = np.column_stack([
                data[index_current, 0],
                data[index_current, order[0]],
                data[index_current, order[1]],
            ])
        
        print('Load dataset from csv file done.')
        return person_data, frame_data

    def get_agents_from_dataset(self, dataset):
        """
        使用数据计算social关系，并组织为`Agent_part`类或`Frame`类
            return: agents, original_sample_number
        """
        base_path = dir_check(os.path.join('./dataset_npz/', '{}'.format(dataset)))
        npy_path = self.npy_file_base_path.format(dataset)

        if os.path.exists(npy_path):
            # 从保存的npy数据集文件中读
            video_neighbor_list, video_matrix, frame_list = self.load_video_matrix(dataset)
        else:
            # 新建npy数据集文件
            person_data, frame_data = self.data_loader(dataset)
            video_neighbor_list, video_matrix, frame_list = self.create_video_matrix(
                person_data, 
                frame_data, 
                save_path=npy_path
            )

        if self.args.train_base == 'agent':
            data_manager = self.get_agents(video_neighbor_list, video_matrix, frame_list)
            print('\nPrepare agent data in dataset {} done.'.format(dataset))
            return data_manager
        
    def load_video_matrix(self, dataset):
        """
        从保存的文件中读取social matrix和social neighbor
        """
        print('Load data from "{}"...'.format(self.npy_file_base_path.format(dataset)))
        all_data = np.load(self.npy_file_base_path.format(dataset), allow_pickle=True)
        video_neighbor_list = all_data['video_neighbor_list']
        video_matrix = all_data['video_matrix']
        frame_list = all_data['frame_list']
        return video_neighbor_list, video_matrix, frame_list

    def create_video_matrix(self, person_data, frame_data, save_path='null'):
        """
        计算social neighbor
        `video_matrix`: shape = [frame_number, person_number, 2]
        """
        person_list = np.sort(np.stack([float(person) for person in person_data])).astype(np.str)
        frame_list = np.sort(np.stack([float(frame) for frame in frame_data])).astype(np.str)

        person_number = len(person_list)
        frame_number = len(frame_list)

        video_matrix = self.args.init_position * np.ones([frame_number, person_number, 2])
        for person in person_data:
            person_index = np.where(person_list == person)[0][0]
            frame_list_current = (person_data[person]).T[0].astype(np.str)
            frame_index_current = np.reshape(np.stack([np.where(frame_current == frame_list) for frame_current in frame_list_current]), [-1])
            traj_current = person_data[person][:, 1:]
            video_matrix[frame_index_current, person_index, :] = traj_current

        video_neighbor_list = []
        for frame_index, data in enumerate(tqdm(video_matrix, desc='Calculate social matrix')):
            person_appear = np.where(np.not_equal(data.T[0], self.args.init_position))[0]
            video_neighbor_list.append(person_appear)

        if not save_path == 'null':
            np.savez(
                save_path, 
                video_neighbor_list=video_neighbor_list,
                video_matrix=video_matrix,
                frame_list=frame_list,
            )

        return video_neighbor_list, video_matrix, frame_list

    def sample_data(self, data_manager, person_index, add_noise=False, reverse=False, desc='Calculate agent data', use_time_bar=True, random_sample=False):
        """
        Sample training data from data_manager
        return: a list of Agent_Part
        """
        agents = []
        if person_index == 'all':
            if random_sample > 0:
                if USE_SEED:
                    random.seed(SEED)
                person_index = random.sample(
                    [i for i in range(data_manager.person_number)], 
                    int(data_manager.person_number * random_sample),
                )
            else:
                person_index = range(data_manager.person_number)

        if use_time_bar:
            itera = tqdm(person_index, desc=desc)
        else:
            itera = person_index

        for person in itera:
            agent_current = data_manager.agent_data[person]
            start_frame = agent_current.start_frame
            end_frame = agent_current.end_frame

            for frame_point in range(start_frame, end_frame, self.args.step):
                if frame_point + self.total_frames > end_frame:
                    break
                
                # type: Agent_Part
                sample_agent = data_manager.get_trajectory(
                    person,
                    frame_point, 
                    frame_point+self.obs_frames, 
                    frame_point+self.total_frames,
                    future_interaction=self.args.future_interaction,
                    calculate_social=self.args.calculate_social,
                    normalization=self.args.normalization,
                    add_noise=add_noise,
                    reverse=reverse,
                )     
                agents.append(sample_agent)
        return agents
    
    def get_agents(self, video_neighbor_list, video_matrix, frame_list):
        """
        使用social matrix计算每个人的`Agent`类，并取样得到用于训练的`Agent_part`类数据
            return: agents(取样后, type=`Agent_part`), original_sample_number
        """
        data_manager = DataManager(
            video_neighbor_list, video_matrix, frame_list, self.args.init_position
        )
        return data_manager

        agents = self.sample_data(data_manager)
        original_sample_number = len(agents)

        if self.args.reverse:
            agents += self.sample_data(data_manager, reverse=True, desc='Preparing reverse data')

        if self.args.add_noise:                
            for repeat in tqdm(range(self.args.add_noise), desc='Preparing noise data'):
                agents += self.sample_data(data_manager, add_noise=True, use_time_bar=False)

        return agents, original_sample_number


class DataManager():
    def __init__(self, video_neighbor_list, video_matrix, frame_list, init_position):
        self.video_neighbor_list = video_neighbor_list
        self.video_matrix = video_matrix
        self.frame_list = frame_list
        self.init_position = init_position
        self.agent_data = self.prepare_agent_data()

    def prepare_agent_data(self):
        self.frame_number, self.person_number, _ = self.video_matrix.shape
        agent_data = []
        for person in range(self.person_number):
            agent_data.append(Agent(
                person, 
                self.video_neighbor_list, 
                self.video_matrix, 
                self.frame_list,
                self.init_position,
            ))
        return agent_data

    def get_trajectory(self, agent_index, start_frame, obs_frame, end_frame, future_interaction=True, calculate_social=True, normalization=False, add_noise=False, reverse=False):
        target_agent = self.agent_data[agent_index]
        frame_list = target_agent.frame_list
        neighbor_list = target_agent.video_neighbor_list[obs_frame-1].tolist()
        neighbor_list = set(neighbor_list) - set([agent_index])
        neighbor_agents = [self.agent_data[nei] for nei in neighbor_list]

        return Agent_Part(
            target_agent, neighbor_agents, frame_list, start_frame, obs_frame, end_frame, future_interaction=future_interaction, calculate_social=calculate_social, normalization=normalization, add_noise=add_noise, reverse=reverse
        )
        

class Agent():
    def __init__(self, agent_index, video_neighbor_list, video_matrix, frame_list, init_position):
        self.agent_index = agent_index
        self.traj = video_matrix[:, agent_index, :]
        self.video_neighbor_list = video_neighbor_list
        self.frame_list = frame_list

        self.start_frame = np.where(np.not_equal(self.traj.T[0], init_position))[0][0]
        self.end_frame = np.where(np.not_equal(self.traj.T[0], init_position))[0][-1] + 1    # 取不到


class Agent_Part():
    def __init__(self, target_agent, neighbor_agents, frame_list, start_frame, obs_frame, end_frame, future_interaction=True, calculate_social=True, normalization=False, add_noise=False, reverse=False):        
        # Trajectory info
        self.start_frame = start_frame
        self.obs_frame = obs_frame
        self.end_frame = end_frame
        self.obs_length = obs_frame - start_frame
        self.total_frame = end_frame - start_frame
        self.frame_list = frame_list[start_frame:end_frame]
        self.vertual_agent = False

        # Trajectory
        self.traj = target_agent.traj[start_frame:end_frame]
        if add_noise:
            noise_curr = np.random.normal(0, 0.1, size=self.traj.shape)
            self.traj += noise_curr
            self.vertual_agent = True

        elif reverse:
            self.traj = self.traj[::-1]
            self.vertual_agent = True
            
        self.pred = 0
        self.start_point = self.traj[0]

        # Options
        self.future_interaction = future_interaction
        self.calculate_social = calculate_social  
        self.normalization = normalization 

        # Neighbor info
        if not self.vertual_agent:
            self.neighbor_traj = []
            for neighbor in neighbor_agents:
                neighbor_traj = neighbor.traj[start_frame:obs_frame]
                neighbor_traj[0:np.maximum(neighbor.start_frame, start_frame)-start_frame] = neighbor_traj[np.maximum(neighbor.start_frame, start_frame)-start_frame]
                neighbor_traj[np.minimum(neighbor.end_frame, obs_frame)-start_frame:obs_frame-start_frame] = neighbor_traj[np.minimum(neighbor.end_frame, obs_frame)-start_frame-1]
                self.neighbor_traj.append(neighbor_traj)
            
            self.neighbor_number = len(neighbor_agents)

        # Initialize
        self.need_to_fix = False
        self.need_to_fix_neighbor = False
        self.initialize()  
        if normalization:
            self.agent_normalization()   

    def initialize(self):
        self.traj_train = self.traj[:self.obs_length]
        self.traj_gt = self.traj[self.obs_length:]

        if self.future_interaction:
            self.traj_pred = predict_linear_for_person(self.traj_train, self.total_frame)[self.obs_length:]
            # self.traj_pred = np.concatenate([self.traj_train, self.traj_future_predict], axis=0)

    def agent_normalization(self):
        """Attention: This method will change the value inside the agent!"""
        self.start_point = np.array([0.0, 0.0])
        if np.linalg.norm(self.traj[0] - self.traj[7]) >= 0.2:
            self.start_point = self.traj[7]
            self.traj = self.traj - self.start_point

        for neighbor_index in range(self.neighbor_number):
            self.neighbor_traj[neighbor_index] -= self.start_point
        
        self.initialize()
        self.need_to_fix = True
        self.need_to_fix_neighbor = True

    def pred_fix(self):
        if not self.need_to_fix:
            return
        
        self.traj += self.start_point
        self.pred += self.start_point

        self.need_to_fix = False
        self.initialize()

    def pred_fix_neighbor(self, pred):
        if not self.need_to_fix_neighbor:
            return pred
            
        pred += self.start_point
        self.need_to_fix_neighbor = False
        return pred

    def get_train_traj(self):
        return self.traj_train

    def get_neighbor_traj(self):
        return self.neighbor_traj

    def get_gt_traj(self):
        return self.traj_gt

    def get_pred_traj(self):
        return self.pred

    def get_pred_traj_sr(self):
        return self.pred_sr

    def get_pred_traj_neighbor(self):
        return self.neighbor_pred

    def write_pred(self, pred):
        self.pred = pred
        self.pred_fix()

    def write_pred_sr(self, pred):
        self.pred_sr = pred
        self.SR = True

    def write_pred_neighbor(self, pred):
        self.neighbor_pred = self.pred_fix_neighbor(pred)

    def calculate_loss(self, loss_function=calculate_ADE_FDE_numpy, SR=False):
        if SR:
            self.loss = loss_function(self.get_pred_traj_sr(), self.get_gt_traj())
        else:
            self.loss = loss_function(self.get_pred_traj(), self.get_gt_traj())
        return self.loss

    def clear_pred(self):
        self.pred = 0

    def draw_results(self, log_dir, file_name, draw_neighbors=False, draw_sr=False):
        save_base_dir = dir_check(os.path.join(log_dir, 'test_figs/'))
        save_format = os.path.join(save_base_dir, file_name)

        obs = self.get_train_traj()
        gt = self.get_gt_traj()
        pred = self.get_pred_traj()

        plt.figure()
        plt.plot(pred.T[0], pred.T[1], '-b*')
        plt.plot(gt.T[0], gt.T[1], '-ro')
        plt.plot(obs.T[0], obs.T[1], '-go')
        if draw_sr:
            pred_sr = self.get_pred_traj_sr()
            plt.plot(pred_sr.T[0], pred.T[1], '--b*')

        if draw_neighbors:
            obs_nei = self.get_neighbor_traj()
            pred_nei = self.get_pred_traj_neighbor()
            
            for obs_c, pred_c in zip(obs_nei, pred_nei):
                plt.plot(pred_c.T[0], pred_c.T[1], '--b*')
                plt.plot(obs_c.T[0], obs_c.T[1], '--go')
        
        plt.axis('scaled')
        plt.title('frame=[{}, {}]'.format(
            self.start_frame,
            self.end_frame,
        ))
        plt.savefig(save_format)
        plt.close()
        

def calculate_distance_matrix(positions, exp=False):
    """input_shape=[person_number, 2]"""
    person_number = len(positions)
    positions_stack = np.stack([positions for _ in range(person_number)])
    distance_matrix = np.linalg.norm(positions_stack - np.transpose(positions_stack, [1, 0, 2]), ord=2, axis=2)
    if exp:
        distance_matrix = np.exp(-0.2 * distance_matrix)
    return distance_matrix


def prepare_agent_for_test(trajs, obs_frames, pred_frames, normalization=False):
    agent_list = []
    for traj in trajs:
        agent_list.append(Agent_Part(
            traj, 0, 0, 
            0, obs_frames, obs_frames+pred_frames, 
            future_interaction=False, calculate_social=False, normalization=normalization
        ))
    return agent_list
