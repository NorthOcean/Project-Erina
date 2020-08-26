'''
Author: Conghao Wong
Date: 1970-01-01 08:00:00
LastEditors: Conghao Wong
LastEditTime: 2020-08-26 00:10:48
Description: file content
'''

import numpy as np
import tensorflow as tf

class TrajGridMap():
    def __init__(self, agent_list:list):
        self.agent_list = agent_list
        
        self.window_size_expand_meter = 5.0
        self.window_size_grid = 4

        self.grid_map = 'null'
        self.traj = self.get_all_obs_traj()
        self.grid_map, self.W, self.b = self.initialize_grid_map(self.traj)
        self.add_to_grid()


    def get_all_obs_traj(self):
        traj = []
        for agent in self.agent_list:
            if agent.rotate == 0:
                traj.append(agent.get_train_traj())
        return np.stack(traj)

    def initialize_grid_map(self, traj):
        x_max = np.max(traj[:, :, 0])
        x_min = np.min(traj[:, :, 0])
        y_max = np.max(traj[:, :, 1])
        y_min = np.min(traj[:, :, 1])
        grid_map = np.zeros([
            int((x_max - x_min + 2*self.window_size_expand_meter)*self.window_size_grid) + 1,
            int((y_max - y_min + 2*self.window_size_expand_meter)*self.window_size_grid) + 1,
        ])
        
        W = np.array([self.window_size_grid, self.window_size_grid])
        b = np.array([x_min - self.window_size_expand_meter, y_min - self.window_size_expand_meter])

        self.mvalue = [x_max, x_min, y_max, y_min]
        return grid_map, W, b

    def add_to_grid(self, val=1):
        for traj in self.traj:
            grid_pos = self.real2grid(traj)
            self.grid_map[grid_pos.T[0], grid_pos.T[1]] += val

    def real2grid(self, traj:np.array):
        return ((traj - self.b) * self.W).astype(np.int)
        

if __name__ == '__main__':
    agent_list = np.load('./train_agents.npy', allow_pickle=True)
    TrajGridMap(agent_list)