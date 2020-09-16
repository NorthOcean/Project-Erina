'''
Author: Conghao Wong
Date: 2020-09-15 15:21:13
LastEditors: Conghao Wong
LastEditTime: 2020-09-15 20:34:01
Description: file content
'''
import numpy as np

class ToyDataManager():
    def __init__(self, r, sigma_position, v_x, sigma_v_x, agent_number, sample_time=0.4, sample_frame=10):
        self.r, self.sigma_position, self.v_x, self.sigma_v_x, self.agent_number, self.sample_time, self.sample_frame = [r, sigma_position, v_x, sigma_v_x, agent_number, sample_time, sample_frame]

    def generate_half_circle_data(self, save_path='./data/toy/half_circle/true_pos_.csv', positive=True):
        start_point_list = [
            [add_gaussion(np.array([-self.r, 0]), u=0, sigma=self.sigma_position), add_gaussion(self.v_x, u=0, sigma=self.sigma_v_x), np.array([self.r, 0])],
            [add_gaussion(np.array([self.r, 0]), u=0, sigma=self.sigma_position), add_gaussion(-self.v_x, u=0, sigma=self.sigma_v_x), np.array([-self.r, 0])],
        ]

        frame = 0
        all_positions = []
        for agent_index in range(self.agent_number):
            positions = []
            start_position, init_v, destination = random_choose(start_point_list, [0.5, 0.5])
            bias_current = add_gaussion(np.array([0, 0]), 0, self.sigma_position)
    
            p = start_position        
            while np.abs((p[0] - destination[0])**2 + (p[1] - destination[1])**2) >= 1:
                frame += self.sample_frame
                p = self.generate_next_step_halfcircle(p, init_v, positive)
                positions.append(np.array([frame, agent_index, p[0] + bias_current[0], p[1] + bias_current[1]]))
            
            all_positions += positions
                
        all_positions = np.column_stack(all_positions)
        np.savetxt(save_path, all_positions, delimiter=',')

    def generate_next_step_halfcircle(self, p, v, positive=True):
        x_new = p[0] + self.sample_time * add_gaussion(v, 0, self.sigma_v_x)
        if self.r**2 - x_new**2 >= 0:
            if positive:
                y_new = np.sqrt(self.r**2 - x_new**2)
            else:
                y_new = -np.sqrt(self.r**2 - x_new**2)
        else:
            y_new = 0

        p = np.array([x_new, y_new])
        return p

    def generate_lincircle_data(self, save_path='./data/toy/line_circle/true_pos_.csv', positive=True):
        start_point_list = [
            [add_gaussion(np.array([-self.r, 0]), u=0, sigma=self.sigma_position), add_gaussion(self.v_x, u=0, sigma=self.sigma_v_x), np.array([self.r, self.r])],
            [add_gaussion(np.array([self.r, self.r]), u=0, sigma=self.sigma_position), add_gaussion(-self.v_x, u=0, sigma=self.sigma_v_x), np.array([-self.r, 0])],
        ]

        frame = 0
        all_positions = []
        for agent_index in range(self.agent_number):
            positions = []
            start_position, init_v, destination = random_choose(start_point_list, [0.5, 0.5])
            bias_current = add_gaussion(np.array([0, 0]), 0, self.sigma_position)
    
            p = start_position        
            while np.abs((p[0] - destination[0])**2 + (p[1] - destination[1])**2) >= 1:
                frame += self.sample_frame
                p = self.generate_next_step_lincircle(p, init_v, positive)
                positions.append(np.array([frame, agent_index, p[0] + bias_current[0], p[1] + bias_current[1]]))
            
            all_positions += positions
                
        all_positions = np.column_stack(all_positions)
        np.savetxt(save_path, all_positions, delimiter=',')

    def generate_next_step_lincircle(self, p, v, positive=True):
        x_new = p[0] + self.sample_time * add_gaussion(v, 0, self.sigma_v_x)
        
        if x_new < 0:
            if self.r**2 - x_new**2 >= 0:
                y_new = np.sqrt(self.r**2 - x_new**2)
            else:
                y_new = 0
        else:
            y_new = self.r

        if not positive:
            y_new = -y_new

        p = np.array([x_new, y_new])
        return p


def random_choose(choose_list:list, diff_weights=False):
    number = len(choose_list)

    if type(diff_weights) == list:
        diff_weights = np.array(diff_weights)
    
    if not type(diff_weights) == bool:
        pro_value = np.array([np.sum(diff_weights[:index]) for index in range(number)])
        pro_value = pro_value / np.sum(diff_weights)

    else:
        diff_weights = np.ones(number)
        pro_value = np.array([np.sum(diff_weights[:index]) for index in range(number)])
        pro_value = pro_value / np.max(pro_value)

    random_value = np.random.rand()
    diff = random_value - pro_value
    choose_index = np.where(diff >= 0)[0][-1]
    return choose_list[choose_index]


def add_gaussion(original_data, u=0, sigma=1):
    if not type(original_data) in [np.array, np.ndarray]:
        original_data = np.array(original_data)
    shape = original_data.shape
    return original_data + np.random.normal(u, sigma, shape)

def merge_datasets(dataset_path0, dataset_path1, save_path):
    set0 = np.loadtxt(dataset_path0, delimiter=',')
    set1 = np.loadtxt(dataset_path1, delimiter=',')

    set1[1, :] += (set0[1][-1] + 1)
    new_set = np.concatenate([set0, set1], axis=1)
    new_set[0, :] = 10 * np.arange(new_set.shape[1])
    np.savetxt(save_path, new_set, delimiter=',')


dm = ToyDataManager(
    r=10,
    sigma_position=0.8,
    v_x=1,
    sigma_v_x=0.1,
    agent_number=50,
    sample_time=0.4,
    sample_frame=10,
)

dm.generate_half_circle_data(save_path='./data/toy/half_circle/true_pos_.csv')
# dm.generate_half_circle_data(save_path='./data/toy/half_circle_neg/true_pos_.csv', positive=False)
dm.generate_lincircle_data(save_path='./data/toy/line_circle/true_pos_.csv')
# dm.generate_lincircle_data(save_path='./data/toy/line_circle_neg/true_pos_.csv', positive=False)
merge_datasets('./data/toy/half_circle/true_pos_.csv', './data/toy/line_circle/true_pos_.csv', './data/toy/true_pos_.csv')