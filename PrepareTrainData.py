'''
@Author: ConghaoWong
@Date: 2019-12-20 09:39:02
LastEditors: Conghao Wong
LastEditTime: 2020-08-15 22:11:38
@Description: file content
'''
import os
import random
USE_SEED = True
SEED = 10

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
        
        all_data = []
        sample_number_original = 0
        train_data = []
        for dataset in train_list:
            data_current, number_current = self.get_agents_from_dataset(dataset)
            all_data += data_current
            sample_number_original += number_current
            train_data.append(data_current)
        
        sample_number_total = len(all_data)
        sample_time = int(sample_number_total / sample_number_original)
        
        if self.args.train_type == 'one':
            index = set([i for i in range(sample_number_original)])
            if USE_SEED:
                random.seed(SEED)
            train_index = random.sample(index, int(sample_number_original * self.args.train_percent))
            test_index = list(index - set(train_index))
            
            train_data = [all_data[(more_sample + 1) * index] for more_sample in range(sample_time) for index in train_index]
            train_index = [(more_sample + 1) * index for more_sample in range(sample_time) for index in train_index]
            
            test_data = [all_data[index] for index in test_index]
            test_index = test_index
        
        elif self.args.train_type == 'all':
            train_data = all_data
            test_data, _ = self.get_agents_from_dataset(test_list[0])
        
        train_info = dict()
        # train_info['all_agents'] = all_data
        train_info['train_data'] = train_data
        train_info['test_data'] = test_data
        train_info['train_number'] = len(train_data)
        train_info['sample_time'] = sample_time  

        # test_options
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
            video_neighbor_list, video_matrix = self.load_video_matrix(dataset)
        else:
            # 新建npy数据集文件
            person_data, frame_data = self.data_loader(dataset)
            video_neighbor_list, video_matrix = self.create_video_matrix(
                person_data, 
                frame_data, 
                save_path=npy_path
            )

        if self.args.train_base == 'agent':
            agents, original_sample_number, self.all_agents = self.get_agents(video_neighbor_list, video_matrix)
            print('\nPrepare agent data in dataset {} done.'.format(dataset))
            return agents, original_sample_number
            
        elif self.args.train_base == 'frame':
            frames, original_sample_number = self.get_frames(video_neighbor_list, video_matrix, self.args.init_position)
            return frames, original_sample_number
        
    def load_video_matrix(self, dataset):
        """
        从保存的文件中读取social matrix和social neighbor
        """
        print('Load data from "{}"...'.format(self.npy_file_base_path.format(dataset)))
        all_data = np.load(self.npy_file_base_path.format(dataset), allow_pickle=True)
        video_neighbor_list = all_data['video_neighbor_list']
        video_matrix = all_data['video_matrix']
        return video_neighbor_list, video_matrix

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
        for frame_index, frame_data in enumerate(tqdm(video_matrix, desc='Calculate social matrix')):
            person_appear = np.where(np.not_equal(frame_data.T[0], self.args.init_position))[0]
            video_neighbor_list.append(person_appear)

        if not save_path == 'null':
            np.savez(
                save_path, 
                video_neighbor_list=video_neighbor_list,
                video_matrix=video_matrix,
            )

        return video_neighbor_list, video_matrix
    
    def get_agents(self, video_neighbor_list, video_matrix):
        """
        使用social matrix计算每个人的`Agent`类，并取样得到用于训练的`Agent_part`类数据
            return: agents(取样后, type=`Agent_part`), original_sample_number, all_agents(type=`Agent`)
        """
        frame_number, person_number, _ = video_matrix.shape
        all_agents = []
        agents = []
        for person in range(person_number):
            all_agents.append(Agent(
                person, 
                video_neighbor_list, 
                video_matrix, 
                self.args.init_position,
            ))

        for person in tqdm(range(person_number), desc='Calculate agent data'):
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
                    all_agents=all_agents,
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
            if self.args.noise_on_reverse:
                current_sample_number = len(agents)
            else:
                current_sample_number = original_sample_number
                
            for repeat in tqdm(range(self.args.add_noise), desc='Preparing data with noise'):
                for index in range(current_sample_number):
                    agents.append(agents[index].add_noise(u=0, sigma=0.1))
        
        return agents, original_sample_number, all_agents

    def get_frames(self, video_neighbor_list, video_matrix, init_position):
        """
        取样得到用于训练的`Frames`类数据
            return: frames(取样后, type=`Frames`), original_sample_number
        """
        frame_number, _, _ = video_matrix.shape
        frames = []
        for frame_point in tqdm(range(0, frame_number, self.args.step), desc='Calculate frame data'):
            if frame_point + self.args.obs_frames + self.args.pred_frames >= frame_number:
                break

            frames.append(Frame(
                frame_point, self.args.obs_frames, self.args.pred_frames,
                video_neighbor_list,
                video_matrix, 
                init_position
            ))
        original_sample_number = len(frames)

        if self.args.reverse:
            for index in tqdm(range(original_sample_number), desc='Preparing reverse data'):
                frames.append(frames[index].reverse())

        if self.args.add_noise:
            # print('Preparing data with noise...')
            if self.args.noise_on_reverse:
                current_sample_number = len(frames)
            else:
                current_sample_number = original_sample_number
                
            for repeat in tqdm(range(self.args.add_noise), desc='Preparing data with noise'):
                for index in range(current_sample_number):
                    frames.append(frames[index].add_noise())

        return frames, original_sample_number

class Agent_Part():
    def __init__(self, agent_index, all_agents, neighbor_list, start_frame, obs_frame, end_frame, future_interaction=True, calculate_social=True, normalization=False):
        self.traj = all_agents[agent_index].traj[start_frame:end_frame]
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
        self.neighbor_traj = []
        for neighbor_index in self.neighbor_list:
            neighbor = all_agents[neighbor_index]
            neighbor_traj = neighbor.traj[start_frame:obs_frame]
            
            neighbor_traj[0:np.maximum(neighbor.start_frame, start_frame)-start_frame] = neighbor_traj[np.maximum(neighbor.start_frame, start_frame)-start_frame]
            neighbor_traj[np.minimum(neighbor.end_frame, obs_frame)-start_frame:obs_frame-start_frame] = neighbor_traj[np.minimum(neighbor.end_frame, obs_frame)-start_frame-1]
            self.neighbor_traj.append(neighbor_traj)
        
        self.neighbor_number = len(self.neighbor_list)
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

    def reverse(self):
        traj_r = self.traj[::-1]
        neighbor_list_r = self.neighbor_list[::-1]
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
            self.start_frame,
            self.obs_frame,
            self.end_frame,
            self.future_interaction
        )
    
    def agent_normalization(self):
        """Attention: This method will change the value inside the agent!"""
        # self.traj_min = np.min(self.traj, axis=0)
        # self.traj_max = np.max(self.traj, axis=0)
        # self.traj_coe = self.traj_max - self.traj_min

        self.start_point = np.array([0.0, 0.0])
        if np.linalg.norm(self.traj[0] - self.traj[7]) >= 0.2:
            self.start_point = self.traj[7]
            self.traj = self.traj - self.start_point
        
        # self.traj = (self.traj - self.traj_min)/self.traj_coe
        self.initialize()
        self.need_to_fix = True
        self.need_to_fix_neighbor = True

    def pred_fix(self):
        if not self.need_to_fix:
            return
        
        self.traj += self.start_point
        self.pred += self.start_point
        
        # else:
        #     self.traj = self.traj * self.traj_coe + self.traj_min
        #     self.pred = self.pred * self.traj_coe + self.traj_min

        self.need_to_fix = False
        self.initialize()

    def pred_fix_neighbor(self, pred):
        if not self.need_to_fix_neighbor:
            return
            
        pred += self.start_point
        self.need_to_fix_neighbor = False
        return pred

    def get_train_traj(self):
        return [self.traj_train]

    def get_neighbor_traj(self):
        return self.neighbor_traj

    def get_gt_traj(self):
        return self.traj_gt

    def get_pred_traj(self):
        return self.pred

    def write_pred(self, pred):
        self.pred = pred
        self.pred_fix()

    def write_pred_neighbor(self, pred):
        self.neighbor_pred = self.pred_fix_neighbor(pred)

    def calculate_loss(self, loss_function):
        self.loss = loss_function(self.get_pred_traj(), self.get_gt_traj())
        return self.loss

    def clear_pred(self):
        self.pred = 0


class Agent():
    def __init__(self, agent_index, video_neighbor_list, video_matrix, init_position):
        self.agent_index = agent_index
        self.traj = video_matrix[:, agent_index, :]
        self.video_neighbor_list = video_neighbor_list

        self.start_frame = np.where(np.not_equal(self.traj.T[0], init_position))[0][0]
        self.end_frame = np.where(np.not_equal(self.traj.T[0], init_position))[0][-1] + 1    # 取不到
    
    def __call__(self, start_frame, obs_frame, end_frame, all_agents, future_interaction=True, calculate_social=True, normalization=False):
        """end_frame is unreachable"""
        neighbor_list = []
        for index in range(start_frame, obs_frame):
            neighbor_list += self.video_neighbor_list[index].tolist()
        neighbor_list = set(neighbor_list) - set([self.agent_index])

        return Agent_Part(
            self.agent_index,
            all_agents, 
            neighbor_list, 
            start_frame, 
            obs_frame, 
            end_frame, 
            future_interaction=future_interaction, 
            calculate_social=calculate_social, 
            normalization=normalization
        )


class Frame():
    def __init__(self, frame_index, obs_length, pred_length, video_neighbor_list, video_matrix, init_position, reverse=False, add_noise=False):
        [self.frame_index, self.obs_length, self.pred_length, self.video_neighbor_list, self.video_matrix, self.init_position] = [frame_index, obs_length, pred_length, video_neighbor_list, video_matrix, init_position]
        
        self.total_length = obs_length + pred_length
        self.global_reverse = reverse
        self.global_noise = add_noise
        self.start_frame = frame_index
        self.end_frame = frame_index + obs_length + pred_length
        self.pred = []
        self.traj_original = video_matrix[frame_index:frame_index+self.total_length, :, :]

        self.vaild_person_index = np.where(np.not_equal(np.mean(self.traj_original[:, :, 0], axis=0), init_position))[0]
        self.vaild_person_number = len(self.vaild_person_index)
        self.traj = np.transpose(self.traj_original[:, self.vaild_person_index, :], [1, 0, 2])

        self.traj_fix = self.traj_fix()
        self.traj_obs = self.traj_fix[:, :obs_length]
        self.traj_gt = self.traj_fix[:, -pred_length:]

    def traj_fix_one(self, traj, u=0, sigma=0.1):
        """
        对于轨迹部分缺失的人，使用加权最小二乘补全
        """
        fix_length = np.sum(traj[:, 0] == self.init_position)

        if self.global_reverse:
            traj = traj[::-1]
            
        if fix_length == 0:
            return traj

        reverse_flag = False
        if traj[0, 0] == self.init_position:
            traj = traj[::-1]
            reverse_flag = True

        if len(traj) - fix_length >= 2:
            traj_pred = predict_linear_for_person(traj[:-fix_length], len(traj), different_weights=0.95)[-fix_length:]
            traj_fix = np.concatenate([traj[:-fix_length], traj_pred], axis=0)
        
        else:
            traj_fix = np.stack([traj[0] for _ in range(len(traj))])

        if self.global_noise:
            noise_curr = np.random.normal(u, sigma, size=traj_fix.shape)
            traj_fix = traj_fix + noise_curr
        
        if reverse_flag:
            return traj_fix[::-1]
        else:
            return traj_fix

    def traj_fix(self):
        traj_fix = np.stack([
            self.traj_fix_one(traj) for traj in self.traj
        ])
        return traj_fix

    def reverse(self):
        return Frame(
            self.frame_index, 
            self.obs_length, 
            self.pred_length, 
            self.video_neighbor_list,
            self.video_matrix, 
            self.init_position, 
            reverse=True,
        )

    def add_noise(self):
        return Frame(
            self.frame_index, 
            self.obs_length, 
            self.pred_length, 
            self.video_neighbor_list, 
            self.video_matrix, 
            self.init_position, 
            add_noise=True,
        )
    
    def get_train_traj(self):
        return self.traj_obs

    def get_gt_traj(self):
        return self.traj_gt

    def get_pred_traj(self):
        return np.stack(self.pred)

    def write_pred(self, pred, index):
        self.pred[index] = pred

    def clear_pred(self):
        self.pred = [[] for _ in range(len(self.traj_gt))]


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
