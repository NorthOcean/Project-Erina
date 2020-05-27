'''
@Author: Conghao Wong
@Date: 2020-05-25 20:14:28
@LastEditors: Conghao Wong
@LastEditTime: 2020-05-25 20:50:59
@Description: file content
'''

import argparse
import numpy as np
from PrepareTrainData import Agent, Agent_Part

def get_parser():
    parser = argparse.ArgumentParser(description='GirdRefinement'')

    # basic settings
    parser.add_argument('--obs_frames', type=int, default=8)
    parser.add_argument('--pred_frames', type=int, default=12)
    parser.add_argument('--test_set', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=1)

    
    return parser

class GirdMap():
    def __init__(self, args):
        self.args = args

        self.create_gird_map()

    def create_gird_map():
        self.map = np.zeros([self.args.gird_shape_x, self.args.gird_shape_y])
