'''
@Author: Conghao Wong
@Date: 2020-05-25 20:14:28
@LastEditors: Conghao Wong
@LastEditTime: 2020-06-01 14:37:20
@Description: file content
'''

import argparse

import cv2
import numpy as np
import tensorflow as tf
from scipy import signal
from tensorflow import keras
from tqdm import tqdm

from PrepareTrainData import Frame

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_parser():
    parser = argparse.ArgumentParser(description='GirdRefinement')

    # basic settings
    parser.add_argument('--obs_frames', type=int, default=8)
    parser.add_argument('--pred_frames', type=int, default=12)
    
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

        self.gird_map = []       
        for index in tqdm(range(self.person_number)):
            self.gird_map.append(self.create_gird_map(index, self.person_number, save=True))
        print('!')
    
    def __calculate_gird(self, input_coor):
        new_coor = [
            int(input_coor[0]//self.args.gird_length + self.args.gird_shape_x//2), 
            int(input_coor[1]//self.args.gird_length + self.args.gird_shape_y//2),
        ]
        return new_coor

    def real2gird(self, real_coor):
        return np.stack([self.__calculate_gird(coor) for coor in real_coor])

    def create_gird_map(self, current_person_index, person_number, save=False):
        mmap = np.zeros([self.args.gird_shape_x, self.args.gird_shape_y])
        pred_current = self.pred_original[current_person_index]
        gird_coor_list = self.real2gird(pred_current)

        other_person_list = [i for i in range(person_number)]# if not i == current_person_index]
        
        for other_person_index in other_person_list:
            pred = self.pred_original[other_person_index]
            coor = self.real2gird(pred)
            mmap = self.add_to_gird(
                coor, 
                mmap,
                calculate_cosine(
                    pred_current[-1] - pred_current[0], 
                    pred[-1] - pred[0]
                ),
            )
        
        mmap_smooth = self.smooth_gird(mmap)
        mmap_new = self.add_to_gird(gird_coor_list, mmap_smooth)
        if save:
            cv2.imwrite(
                './gird_{}.png'.format(current_person_index),
                127*(mmap_new/mmap_new.max()+1)
            )
        return mmap_smooth
    
    def add_to_gird(self, coor_list, girdmap, coe=1):
        coor_list_new = []  # 删除重复项目
        for coor in coor_list:
            if not coor.tolist() in coor_list_new:
                coor_list_new.append(coor.tolist())

        for coor in coor_list_new:
            girdmap[coor[0], coor[1]] += coe
        return girdmap

    def smooth_gird(self, girdmap):
        return signal.convolve2d(girdmap, self.kernel, 'same')


class SR(tf.keras.layers.Layer):
    def __init__(self):
        super(SR, self).__init__()
    
    def build(self, input_shape):
        self.bias = self.add_variable('kernel', shape=[input_shape[-1]])

    def call(self, inputs):
        return inputs + self.bias


class SocialRefine():
    def __init__(self, args):
        self.args = args

    def train_model(self, input_traj, gird_map, epochs=30):
        



def calculate_cosine(vec1, vec2):
    """
    两个输入均为表示方向的向量, shape=[2]
    """
    length1 = np.linalg.norm(vec1)
    length2 = np.linalg.norm(vec2)
    return np.sum(vec1 * vec2) / (length1 * length2)


if __name__ == "__main__":
    args = get_parser().parse_args()
    frame_data = np.load('./testframes.npy', allow_pickle=True)
    a = GirdMap(args, frame_data[13])
    sr = SocialRefine(args)
    
    for i in range(9):
        output_pred = sr.train_model(a, i)
        original_gird_map = a.gird_map[i]
        original_pred = a.real2gird(a.pred_original[i])
        # np.savetxt('res.txt', output)

        mmap_original = a.add_to_gird(original_pred, original_gird_map)
        mmap_res = a.add_to_gird(output_pred, original_gird_map)
        cv2.imwrite(
            './gird_original_{}.png'.format(i),
            127*(mmap_original/mmap_original.max()+1)
        )

        cv2.imwrite(
            './gird_res_{}.png'.format(i),
            127*(mmap_res/mmap_res.max()+1)
        )

    print('!')
