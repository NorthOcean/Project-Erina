'''
Author: Conghao Wong
Date: 1970-01-01 08:00:00
LastEditors: Conghao Wong
LastEditTime: 2020-08-26 00:39:52
Description: file content
'''

import numpy as np
import cv2

def savemap(npy, target):
    gm = np.load(npy)
    gm *= 255/np.max(gm)
    cv2.imwrite(target, gm.astype(np.int))

for index in [0, 1, 2, 3, 4]:
    savemap('./gridmaps/{}n20.npy'.format(index), './gridmaps/gm{}-n20.png'.format(index))
