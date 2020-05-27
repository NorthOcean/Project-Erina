'''
@Author: Conghao Wong
@Date: 2020-05-25 20:14:28
@LastEditors: Conghao Wong
@LastEditTime: 2020-05-27 14:46:54
@Description: file content
'''

import cv2
import argparse
import numpy as np
from tqdm import tqdm
from scipy import signal
from PrepareTrainData import Frame

def get_parser():
    parser = argparse.ArgumentParser(description='GirdRefinement')

    # basic settings
    parser.add_argument('--gird_shape_x', type=int, default=150)
    parser.add_argument('--gird_shape_y', type=int, default=150)
    parser.add_argument('--gird_length', type=float, default=0.3)   # 网格的真实长度
    parser.add_argument('--smooth_size', type=int, default=8)

    return parser

class GirdMap():
    def __init__(self, args, frame_data:Frame):
        self.args = args
        self.frame_data = frame_data

        self.kernel = np.ones([self.args.smooth_size, self.args.smooth_size])/self.args.smooth_size**2
        self.pred_original = frame_data.get_pred_traj()
        self.person_number = frame_data.vaild_person_number
        self.person_list = frame_data.vaild_person_index
        
        for index in tqdm(range(self.person_number)):
            self.create_gird_map(index, self.person_number)
        print('!')

    def create_gird_map(self, current_person_index, person_number):
        mmap = np.zeros([self.args.gird_shape_x, self.args.gird_shape_y])
        other_person_list = [i for i in range(person_number) if not i == current_person_index]
        direction_vec = []
        
        for other_person_index in other_person_list:
            pred_current = self.pred_original[other_person_index]
            gird_coor_list = [self.calculate_gird(coor) for coor in pred_current]
            mmap = self.add_to_gird(gird_coor_list, mmap)
        
        mmap_smooth = self.smooth_gird(mmap)
        current_predict = self.pred_original[current_person_index]
        gird_coor_list = [self.calculate_gird(coor) for coor in current_predict]
        mmap_new = self.add_to_gird(gird_coor_list, mmap_smooth)
        cv2.imwrite('./gird_{}.png'.format(current_person_index), 40*mmap_new)

    def calculate_gird(self, input_coor):
        new_coor = [
            int(input_coor[0]//self.args.gird_length + self.args.gird_shape_x//2), 
            int(input_coor[1]//self.args.gird_length + self.args.gird_shape_y//2),
        ]
        return new_coor
    
    def add_to_gird(self, coor_list, girdmap, coe=1):
        coor_list_new = []  # 删除重复项目
        for coor in coor_list:
            if not coor in coor_list_new:
                coor_list_new.append(coor)

        for coor in coor_list_new:
            girdmap[coor[0], coor[1]] += coe
        return girdmap

    def smooth_gird(self, girdmap):
        return signal.convolve2d(girdmap, self.kernel, 'same')


if __name__ == "__main__":
    args = get_parser().parse_args()
    frame_data = np.load('./testframes.npy', allow_pickle=True)
    a = GirdMap(args, frame_data[13])
