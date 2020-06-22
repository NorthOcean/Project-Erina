'''
@Author: Conghao Wong
@Date: 2020-05-25 20:14:28
@LastEditors: Conghao Wong
@LastEditTime: 2020-06-01 21:33:23
@Description: file content
'''

import argparse

import cv2
import numpy as np
import tensorflow as tf
from scipy import signal
from tensorflow import keras
from tqdm import tqdm
import matplotlib.pyplot as plt

from PrepareTrainData import Frame

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def get_parser():
    parser = argparse.ArgumentParser(description='GirdRefinement')

    # basic settings
    parser.add_argument('--obs_frames', type=int, default=8)
    parser.add_argument('--pred_frames', type=int, default=12)
    
    parser.add_argument('--gird_shape_x', type=int, default=700)
    parser.add_argument('--gird_shape_y', type=int, default=700)
    parser.add_argument('--gird_length', type=float, default=0.1)   # 网格的真实长度
    parser.add_argument('--avoid_size', type=int, default=10)   # 主动避让的半径网格尺寸
    parser.add_argument('--social_size', type=int, default=1)   # 互不侵犯的半径网格尺寸
    parser.add_argument('--smooth_size', type=int, default=5)   # 进行平滑的窗口网格边长
    parser.add_argument('--max_refine', type=float, default=1.0)   # 最大修正尺寸

    parser.add_argument('--savefig', type=int, default=True)

    return parser

class GirdMap():
    def __init__(self, args, frame_data:Frame):
        self.args = args
        self.frame_data = frame_data

        self.kernel = np.ones([self.args.smooth_size, self.args.smooth_size])/self.args.smooth_size**2
        self.pred_original = frame_data.get_pred_traj()
        self.person_number = frame_data.vaild_person_number
        self.person_list = frame_data.vaild_person_index

        self.add_mask = np.zeros([101, 101])
        for i in range(101):
            for j in range(101):
                self.add_mask[i, j] = ((i-50)**2 + (j-50)**2)**0.5
        
        self.add_mask = np.zeros([101, 101]) * (self.add_mask > 50) + (50 - self.add_mask) * (self.add_mask <= 50)

        self.gird_map = []       
        for index in range(self.person_number):
            self.gird_map.append(self.create_gird_map(index, self.person_number, save=True))
        # print('!')
    
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

        other_person_list = [i for i in range(person_number) if not i == current_person_index]

        avoid_person_list = []
        avoid_person_cosine = []
        interest_person_list = []
        interest_person_cosine = []
        for other_person_index in other_person_list:
            pred = self.pred_original[other_person_index]
            cosine = calculate_cosine(pred_current[-1] - pred_current[0], pred[-1] - pred[0])
            if cosine >= 0:
                interest_person_list.append(other_person_index)
                interest_person_cosine.append(cosine)
            else:
                avoid_person_list.append(other_person_index)
                avoid_person_cosine.append(cosine)
        
        mask = np.stack([(self.args.pred_frames-i)/self.args.pred_frames for i in range(self.args.pred_frames)]).reshape([-1, 1])
        
        mmap = self.add_to_gird(
            gird_coor_list, 
            mmap,
            coe=-1 * np.ones_like(mask),
            add_size=self.args.avoid_size,
        )
        
        for other_person_index in other_person_list:
            pred = self.pred_original[other_person_index]
            coor = self.real2gird(pred)
            mmap = self.add_to_gird(
                coor, 
                mmap,
                coe=mask * np.maximum(-1.0 * calculate_cosine(
                    pred_current[-1] - pred_current[0], 
                    pred[-1] - pred[0]
                ), 0),
                add_size=self.args.avoid_size,
            )
            # mmap = self.add_to_gird(
            #     coor, 
            #     mmap,
            #     coe=mask * np.maximum(calculate_cosine(
            #         pred_current[-1] - pred_current[0], 
            #         pred[-1] - pred[0]
            #     ), 0),
            #     add_size=self.args.social_size,
            # )
        
        mmap_smooth = self.smooth_gird(mmap)
        
        mmap_new = self.add_to_gird(gird_coor_list, mmap_smooth, coe=np.ones([self.args.pred_frames]))
        if save:
            cv2.imwrite(
                './gird_{}.png'.format(current_person_index),
                127*(mmap_new/mmap_new.max()+1)
            )
        return mmap_smooth
    
    def add_to_gird(self, coor_list, girdmap, coe=1, add_size=1, interp=True, replace=True):
        mask = cv2.resize(self.add_mask, (2*add_size, 2*add_size))
        girdmap_c = girdmap.copy()
        coor_list_new = []  # 删除重复项目
        for coor in coor_list:
            if not coor.tolist() in coor_list_new:
                coor_list_new.append(coor.tolist())
        
        # coor_list_new = np.stack(coor_list_new)
        if interp:
            coe_new = []
            for i in range(1, len(coor_list_new)):
                if abs(coor_list_new[i][0] - coor_list_new[i-1][0]) + abs(coor_list_new[i][1] - coor_list_new[i-1][1]) <= 1:
                    continue

                for inter_x in range(1, abs(coor_list_new[i][0] - coor_list_new[i-1][0])):
                    coor_list_new.append([coor_list_new[i-1][0]+inter_x, coor_list_new[i-1][1]])
                    coe_new.append(coe[i-1])

                for inter_y in range(1, abs(coor_list_new[i][1] - coor_list_new[i-1][1])):
                    coor_list_new.append([coor_list_new[i][0], coor_list_new[i-1][1]+inter_y])
                    coe_new.append(coe[i-1])

            if len(coe_new):
                coe = np.concatenate([coe, np.stack(coe_new)])
                    

        for coor, coe_c in zip(coor_list_new, coe):
            girdmap_c[coor[0]-add_size:coor[0]+add_size, coor[1]-add_size:coor[1]+add_size] = coe_c*mask + girdmap_c[coor[0]-add_size:coor[0]+add_size, coor[1]-add_size:coor[1]+add_size]
        return girdmap_c

    def smooth_gird(self, girdmap):
        # return signal.convolve2d(girdmap, self.kernel, 'same')
        return girdmap

    def refine_model(self, input_traj, gird_map, epochs=30):
        prev_result = input_traj
        
        if calculate_length(prev_result[-1] - prev_result[1]) <= 1.0:
            return prev_result
            
        for epoch in range(epochs):
            result = prev_result
            input_traj_gird = (self.real2gird(result) + 1.0).astype(np.int)

            diff_x = gird_map[1:, 1:] - gird_map[:-1, 1:]
            diff_y = gird_map[1:, 1:] - gird_map[1:, :-1]

            dx_current = diff_x[input_traj_gird.T[0], input_traj_gird.T[1]]
            dy_current = diff_y[input_traj_gird.T[0], input_traj_gird.T[1]]

            x_bias = dx_current * 0.001
            y_bias = dy_current * 0.001

            prev_result = np.stack([
                result.T[0] - x_bias,
                result.T[1] - y_bias,
            ]).T

        delta = np.maximum(prev_result - input_traj, self.args.max_refine)
        coe = 0.8*np.stack([i/(self.args.pred_frames-1) for i in range(self.args.pred_frames)]).reshape([-1, 1])
    
        # print('!')
        # np.savetxt('res.txt', prev_result)
        # np.savetxt('inp.txt', input_traj)
        return input_traj + coe * delta

        



def calculate_cosine(vec1, vec2):
    """
    两个输入均为表示方向的向量, shape=[2]
    """
    length1 = np.linalg.norm(vec1)
    length2 = np.linalg.norm(vec2)
    return np.sum(vec1 * vec2) / (length1 * length2)


def calculate_length(vec1):
    """
    表示方向的向量, shape=[2]
    """
    length1 = np.linalg.norm(vec1)
    return length1


if __name__ == "__main__":
    args = get_parser().parse_args()
    frame_data = np.load('./testframes.npy', allow_pickle=True)
    savefig = args.savefig
    
    ade_gain = []
    # timebar = tqdm(range(len(frame_data)))
    timebar = tqdm(range(13, 14))
    for frame_index, frame in enumerate(timebar):
        a = GirdMap(args, frame_data[frame])

        traj_refine = []
        for index in range(a.person_number):
            traj_refine.append(a.refine_model(a.pred_original[index], a.gird_map[index], epochs=25))

        frame_data[frame_index].pred_fix = traj_refine
        
        pred_old = frame_data[frame_index].pred
        gt = frame_data[frame_index].traj_gt

        ade_old = []
        ade_new = []
        for pred_old_c, pred_new_c, gt_c in zip(pred_old, traj_refine, gt):
            ade_old.append(np.mean(np.linalg.norm(pred_old_c - gt_c, axis=1)))
            ade_new.append(np.mean(np.linalg.norm(pred_new_c - gt_c, axis=1)))
        
        ade_old = np.mean(np.stack(ade_old))
        ade_new = np.mean(np.stack(ade_new))
        timebar.set_postfix({
            'ade_old':ade_old,
            'ade_fix':ade_new,
            'gain':ade_new-ade_old,
        })
        ade_gain.append(ade_new-ade_old)

        if savefig:
            plt.figure(figsize=(20, 20))
            for res, ori, gt in zip(traj_refine, a.pred_original, frame_data[frame].traj_gt): 
                plt.plot(ori.T[0], ori.T[1], '-^')
                plt.plot(res.T[0], res.T[1], '--*')
                plt.plot(gt.T[0], gt.T[1], '-o')
                
            plt.axis('scaled')
            plt.title('ade_old={}, ade_fix={}'.format(ade_old, ade_new))
            plt.savefig('./sr{}.png'.format(frame))
            plt.close()

    print(np.mean(np.stack(ade_gain)))
