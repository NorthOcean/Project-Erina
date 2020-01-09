'''
@Author: ConghaoWong
@Date: 2019-12-20 09:39:02
@LastEditors  : ConghaoWong
@LastEditTime : 2020-01-09 20:49:15
@Description: file content
'''
import os
import random

import numpy as np
from tqdm import tqdm

from helpmethods import dir_check, list2array, predict_linear_for_person


class Prepare_Train_Data():
    def __init__(self, args, save=True):
        self.args = args
        self.obs_frames = args.obs_frames
        self.pred_frames = args.pred_frames
        self.total_frames = self.pred_frames + self.obs_frames
        self.step = args.step

        self.max_neighbor = args.max_neighbor
        self.god_position = np.array([args.god_position, args.god_position])
        self.god_past_traj = np.stack([self.god_position for _ in range(self.obs_frames)])
        self.god_future_traj = np.stack([self.god_position for _ in range(self.pred_frames)])

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
            train_list = [index for index in range(5) if not index == self.args.test_set]
            test_list = [self.args.test_set]
            
        train_agents_list = []
        for dataset in train_list:
            train_agents_list.append(self.get_agents_from_dataset(dataset))
        
        if self.args.train_type == 'all':
            test_agents_list = []
            for dataset in test_list:
                test_agents_list.append(self.get_agents_from_dataset(dataset))

        if self.args.train_type == 'one':
            agents = train_agents_list[0][0]
            sample_number_original = train_agents_list[0][1]
            sample_number_total = len(agents)
            sample_time = int(sample_number_total / sample_number_original)

            index = set([i for i in range(sample_number_original)])
            train_index = random.sample(index, int(sample_number_original * self.args.train_percent))
            test_index = list(index - set(train_index))
            
            train_agents = [agents[(more_sample + 1) * index] for more_sample in range(sample_time) for index in train_index]
            train_index = [(more_sample + 1) * index for more_sample in range(sample_time) for index in train_index]
            
            test_agents = [agents[index] for index in test_index]
            test_index = test_index

        elif self.args.train_type == 'all':
            sample_number_original = 0
            train_agents = []
            for [agents, sample_number] in train_agents_list:
                sample_number_original += sample_number
                for agent_current in agents:
                    train_agents.append(agent_current)
            train_index = [i for i in range(len(train_agents))]
            sample_number_total = len(train_agents)
            sample_time = int(sample_number_total / sample_number_original)
            
            test_agents = []
            for [agents, sample_number] in test_agents_list:
                for index, agent_current in enumerate(agents):
                    if index < sample_number / sample_time:
                        test_agents.append(agent_current)
            test_index = [i for i in range(len(test_agents))]
        
        train_info = dict()
        train_info['train_agents'] = train_agents
        train_info['train_index'] = train_index
        train_info['test_agents'] = test_agents
        train_info['test_index'] = test_index
        train_info['train_number'] = len(train_index)
        train_info['sample_time'] = sample_time      

        return train_info

    def data_loader(self, dataset_index):
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
        base_path = dir_check(os.path.join('./dataset_npz/', '{}'.format(dataset)))
        npy_path = self.npy_file_base_path.format(dataset)

        if os.path.exists(npy_path):
            # 从保存的npy数据集文件中读
            video_neighbor_list, video_social_matrix, video_matrix = self.load_video_matrix(dataset)
        else:
            # 新建npy数据集文件
            person_data, frame_data = self.data_loader(dataset)
            video_neighbor_list, video_social_matrix, video_matrix = self.create_video_matrix(
                person_data, 
                frame_data, 
                save_path=npy_path
            )
    
        agents, original_sample_number = self.get_agents(video_neighbor_list, video_social_matrix, video_matrix)
        print('\nPrepare agent data in dataset {} done.'.format(dataset))
        return agents, original_sample_number
        
    def load_video_matrix(self, dataset):
        print('Load data from "{}"...'.format(self.npy_file_base_path.format(dataset)))
        all_data = np.load(self.npy_file_base_path.format(dataset), allow_pickle=True)
        video_neighbor_list = all_data['video_neighbor_list']
        video_social_matrix = all_data['video_social_matrix']
        video_matrix = all_data['video_matrix']
        return video_neighbor_list, video_social_matrix, video_matrix

    def create_video_matrix(self, person_data, frame_data, save_path='null'):
        person_list = np.sort(np.stack([float(person) for person in person_data])).astype(np.str)
        frame_list = np.sort(np.stack([float(frame) for frame in frame_data])).astype(np.str)

        person_number = len(person_list)
        frame_number = len(frame_list)

        video_matrix = self.args.god_position * np.ones([frame_number, person_number, 2])
        for person in person_data:
            person_index = np.where(person_list == person)[0][0]

            frame_list_current = (person_data[person]).T[0].astype(np.str)
            frame_index_current = np.reshape(np.stack([np.where(frame_current == frame_list) for frame_current in frame_list_current]), [-1])
            traj_current = person_data[person][:, 1:]
            video_matrix[frame_index_current, person_index, :] = traj_current

        person_enter_frame = np.where(np.not_equal(video_matrix[:, :, 0], self.args.god_position), 1.0, 0.0).T
        person_enter_frame_smooth = np.ones_like(person_enter_frame)
        n = 3
        for i, person in enumerate(person_enter_frame):
            person_enter_frame_smooth[i, :] = np.convolve(person, np.ones((n))/n, mode='same')

        video_social_matrix = np.zeros([frame_number, person_number, person_number])
        video_neighbor_list = []
        for i, frame in enumerate(tqdm(video_matrix, desc='Calculate social matrix', ncols=300)):
            print('Calculate social matrix in frame {}/{}...'.format(i+1, frame_number), end='\r')
            matrix_half = calculate_distance_matrix(frame, exp=True) * person_enter_frame_smooth[:, i]  # 上三角数据有效
            matrix_total = np.minimum(matrix_half, matrix_half.T)

            neighbor_list = []
            for p, person in enumerate(matrix_total):
                if not np.sum(person):
                    neighbor_list.append([])
                    continue

                neighbor_number = np.sum(person > 0)
                neighbor_list_current = ((np.argsort(person)[-min(self.max_neighbor, neighbor_number):])[::-1]).tolist()
                if (p in neighbor_list) and (self.args.future_interaction == False):
                    neighbor_list_current.remove(p)
                neighbor_list.append(neighbor_list_current)

            video_neighbor_list.append(neighbor_list)
            video_social_matrix[i, :, :] = matrix_total

        if not save_path == 'null':
            np.savez(
                save_path, 
                video_neighbor_list=video_neighbor_list,
                video_social_matrix=video_social_matrix,
                video_matrix=video_matrix,
            )

        return video_neighbor_list, video_social_matrix, video_matrix
    
    def get_agents(self, video_neighbor_list, video_social_matrix, video_matrix):
        frame_number, person_number, _ = video_matrix.shape
        all_agents = []
        agents = []
        for person in range(person_number):
            all_agents.append(Agent(
                person, 
                video_neighbor_list, 
                video_social_matrix, 
                video_matrix, 
                self.args.god_position,
            ))

        for person in tqdm(range(person_number), desc='Calculate agent data'):
            # print('Calculate agent data {}/{}...'.format(person+1, person_number), end='\r')
            agent_current = all_agents[person]
            start_frame = agent_current.start_frame
            end_frame = agent_current.end_frame

            for frame_point in range(start_frame, end_frame, self.args.step):
                if frame_point + self.total_frames > end_frame:
                    break
                
                # type: Agent_Part
                agent_sample = agent_current(
                    frame_point, 
                    frame_point+self.obs_frames, 
                    frame_point+self.total_frames,
                    future_interaction=self.args.future_interaction,
                    calculate_social=self.args.calculate_social,
                    normalization=self.args.normalization,
                )

                if agent_sample.calculate_social:
                    neighbor_list_current = agent_sample.neighbor_list_current
                    for neighbor in neighbor_list_current:
                        neighbor_agent = all_agents[neighbor](frame_point, frame_point+self.obs_frames, frame_point+self.pred_frames)
                        agent_sample.neighbor_agent.append(neighbor_agent)
                        
                agents.append(agent_sample)

        original_sample_number = len(agents)

        if self.args.reverse:
            # print('Preparing reverse data...')
            for index in tqdm(range(original_sample_number), desc='Preparing reverse data'):
                agents.append(agents[index].reverse())

        if self.args.add_noise:
            # print('Preparing data with noise...')
            current_sample_number = len(agents)
            for repeat in tqdm(range(self.args.add_noise), desc='Preparing data with noise'):
                for index in range(current_sample_number):
                    agents.append(agents[index].add_noise(u=0, sigma=0.1))
        
        return agents, original_sample_number


class Agent_Part():
    def __init__(self, traj, neighbor_list, social_vector, start_frame, obs_frame, end_frame, future_interaction=True, calculate_social=True, normalization=False):
        self.traj = traj
        self.pred = 0
        self.start_point = self.traj[0]
        self.start_frame = start_frame
        self.obs_frame = obs_frame
        self.end_frame = end_frame
        self.obs_length = obs_frame - start_frame
        self.total_frame = end_frame - start_frame

        self.future_interaction = future_interaction
        self.calculate_social = calculate_social  
        self.normalization = normalization 

        self.neighbor_list = neighbor_list
        self.social_vector = social_vector
        
        self.already_fixed = False
        self.initialize()  
        if normalization:
            self.agent_normalization()   

    def initialize(self):
        self.traj_train = self.traj[:self.obs_length]
        self.traj_gt = self.traj[self.obs_length:]

        if self.calculate_social:
            self.neighbor_agent = []
            self.neighbor_list_current = self.neighbor_list[self.obs_length]
            self.social_vector_current = self.social_vector[self.obs_length]

        if self.future_interaction:
            self.traj_pred = predict_linear_for_person(self.traj_train, self.total_frame)[self.obs_length:]
            # self.traj_pred = np.concatenate([self.traj_train, self.traj_future_predict], axis=0)

    def reverse(self):
        traj_r = self.traj[::-1]
        neighbor_list_r = self.neighbor_list[::-1]
        social_vector_r = self.social_vector[::-1]
        return Agent_Part(
            traj_r, 
            neighbor_list_r, 
            social_vector_r, 
            self.start_frame, 
            self.obs_frame, 
            self.end_frame, 
            self.future_interaction
        )

    def add_noise(self, u=0, sigma=0.1):
        noise_curr = np.random.normal(u, sigma, size=self.traj.shape)
        traj_noise = self.traj + noise_curr
        return Agent_Part(
            traj_noise,
            self.neighbor_list,
            self.social_vector,
            self.start_frame,
            self.obs_frame,
            self.end_frame,
            self.future_interaction
        )
    
    def agent_normalization(self):
        """Attention: This method will change the value inside the agent!"""
        self.traj_min = np.min(self.traj, axis=0)
        self.traj_max = np.max(self.traj, axis=0)
        self.traj_coe = self.traj_max - self.traj_min
        self.traj = (self.traj - self.traj_min)/self.traj_coe
        # self.start_point = self.traj[0]
        # self.traj = self.traj - self.start_point
        self.initialize()

    def pred_fix(self):
        if self.already_fixed:
            return
        
        self.already_fixed = True
        # self.traj = self.traj + self.start_point * self.normalization
        # self.pred = self.pred + self.start_point * self.normalization
        self.traj = self.traj * self.traj_coe + self.traj_min
        self.pred = self.pred * self.traj_coe + self.traj_min
        self.initialize()


class Agent():
    def __init__(self, agent_index, video_neighbor_list, video_social_matrix, video_matrix, god_position):
        self.traj = video_matrix[:, agent_index, :]
        self.neighbor_list = [neighbor_list[agent_index] for neighbor_list in video_neighbor_list]
        self.social_vector = video_social_matrix[:, agent_index, :]

        self.start_frame = np.where(np.not_equal(self.traj.T[0], god_position))[0][0]
        self.end_frame = np.where(np.not_equal(self.traj.T[0], god_position))[0][-1] + 1    # 取不到
    
    def __call__(self, start_frame, obs_frame, end_frame, future_interaction=True, calculate_social=True, normalization=False):
        """end_frame is unreachable"""
        traj = self.traj[start_frame:end_frame]
        neighbor_list = [self.neighbor_list[frame] for frame in range(start_frame, end_frame)]
        social_vector = self.social_vector[start_frame:end_frame]

        return Agent_Part(
            traj, 
            neighbor_list, 
            social_vector, 
            start_frame, 
            obs_frame, 
            end_frame, 
            future_interaction=future_interaction, 
            calculate_social=calculate_social, 
            normalization=normalization
        )


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
