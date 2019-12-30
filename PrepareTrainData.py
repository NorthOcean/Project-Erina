'''
@Author: ConghaoWong
@Date: 2019-12-20 09:39:02
@LastEditors  : ConghaoWong
@LastEditTime : 2019-12-27 20:03:55
@Description: file content
'''
import numpy as np
import os

from helpmethods import (
    list2array,
    dir_check,
    predict_linear_for_person,
)

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
        self.dataset_save_path = dir_check(os.path.join(dir_check('./dataset_npz/'), '{}/'.format(args.test_set)))
        self.dataset_save_format = os.path.join(self.dataset_save_path, 'data.npz')

        if os.path.exists(self.dataset_save_format):
            video_neighbor_list, video_social_matrix, video_matrix = self.load_video_matrix()
        else:
            person_data, frame_data = self.data_loader()
            video_neighbor_list, video_social_matrix, video_matrix = self.prepare_video_matrix(person_data, frame_data, save=save)
        
        self.train_agents = self.get_agents(video_neighbor_list, video_social_matrix, video_matrix)

    def data_loader(self):
        dataset_index = self.args.test_set
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

    def load_video_matrix(self):
        all_data = np.load(self.dataset_save_format, allow_pickle=True)
        video_neighbor_list = all_data['video_neighbor_list']
        video_social_matrix = all_data['video_social_matrix']
        video_matrix = all_data['video_matrix']
        return video_neighbor_list, video_social_matrix, video_matrix

    def prepare_video_matrix(self, person_data, frame_data, save=True):
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
        for i, frame in enumerate(video_matrix):
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

        if save:
            np.savez(
                self.dataset_save_format, 
                video_neighbor_list=video_neighbor_list,
                video_social_matrix=video_social_matrix,
                video_matrix=video_matrix,
            )

        return video_neighbor_list, video_social_matrix, video_matrix
    
    def get_agents(self, video_neighbor_list, video_social_matrix, video_matrix):
        frame_number, person_number, _ = video_matrix.shape
        all_agents = []
        train_agents = []
        for person in range(person_number):
            all_agents.append(Agent(
                person, 
                video_neighbor_list, 
                video_social_matrix, 
                video_matrix, 
                self.args.god_position, 
                future_interaction=self.args.future_interaction,
                calculate_social=False, # self.args.calculate_social,
                normalization=self.args.normalization,
            ))

        train_agents = []
        train_agents_r = []
        for person in range(person_number):
            print('Prepare agent data {}/{}...'.format(person+1, person_number), end='\r')
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
                )
                neighbor_list_current = agent_sample.neighbor_list_current
                for neighbor in neighbor_list_current:
                    neighbor_agent = all_agents[neighbor](frame_point, frame_point+self.obs_frames, frame_point+self.pred_frames)
                    agent_sample.neighbor_agent.append(neighbor_agent)
                train_agents.append(agent_sample)

        train_samples_number = len(train_agents)

        if self.args.reverse:
            for index in range(train_samples_number):
                train_agents.append(train_agents[index].reverse())

        if self.args.add_noise:
            for repeat in range(self.args.add_noise):
                for index in range(train_samples_number):
                    train_agents.append(train_agents[index].add_noise(u=0, sigma=0.1))
        
        print('Prepare agent data done.\t\t')
        return train_agents


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
        # self.traj = (self.traj - self.traj_min)/self.traj_coe
        self.start_point = self.traj[0]
        self.traj = self.traj - self.start_point
        self.initialize()

    def pred_fix(self):
        if self.already_fixed:
            return
        
        self.already_fixed = True
        self.traj = self.traj + self.start_point * self.normalization
        self.pred = self.pred + self.start_point * self.normalization
        self.initialize()


class Agent():
    def __init__(self, agent_index, video_neighbor_list, video_social_matrix, video_matrix, god_position, future_interaction=True, calculate_social=True, normalization=False):
        self.future_interaction = future_interaction
        self.calculate_social = calculate_social
        self.normalization = normalization

        self.traj = video_matrix[:, agent_index, :]
        self.neighbor_list = [neighbor_list[agent_index] for neighbor_list in video_neighbor_list]
        self.social_vector = video_social_matrix[:, agent_index, :]

        self.start_frame = np.where(np.not_equal(self.traj.T[0], god_position))[0][0]
        self.end_frame = np.where(np.not_equal(self.traj.T[0], god_position))[0][-1] + 1    # 取不到
    
    def __call__(self, start_frame, obs_frame, end_frame):
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
            future_interaction=self.future_interaction, 
            calculate_social=self.calculate_social, 
            normalization=self.normalization
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


        
