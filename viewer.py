'''
@Author: Conghao Wong
@Date: 2020-07-15 19:37:05
@LastEditors: Conghao Wong
@LastEditTime: 2020-07-15 20:35:11
@Description: file content
'''

import argparse
import os
import time
from tkinter import *

import cv2
import numpy as np
from PIL import Image, ImageTk


TIME = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))


class Visualization():
    def __init__(self):
        self.root = Tk()
        self.root.title('Project Erina')
        self.inputs_list = []

        Label(self.root, text='Base Path').grid(row=0, column=0)
        self.path0 = Entry(self.root, width=60)
        self.path0.grid(row=0, column=1, columnspan=4)
        self.path0.insert(0, '/Users/phantom/Project-Erina/feature_vis/traj{}.png')

        Label(self.root, text='Agent 1').grid(row=0, column=5)
        self.path1 = Entry(self.root)
        self.path1.grid(row=0, column=6)

        Label(self.root, text='Agent 2').grid(row=0, column=7)
        self.path2 = Entry(self.root)
        self.path2.grid(row=0, column=8)

        self.ok = Button(self.root, text='OK', command=self.ok_button)
        self.ok.grid(row=0, column=9)

        self.canvas1 = Canvas(self.root, width=640, height=480)   
        self.canvas1.grid(row=1, column=0, columnspan=5)

        self.canvas2 = Canvas(self.root, width=640, height=480)   
        self.canvas2.grid(row=1, column=6, columnspan=5)

        self.root.mainloop()
        cv2.destroyAllWindows()

    def ok_button(self, ):
        path0 = self.path0.get()
        path1 = self.path1.get()
        path2 = self.path2.get()

        img_path1 = path0.format(path1)
        img_path2 = path0.format(path2)

        image1 = Image.open(img_path1)  
        self.image1_tk = ImageTk.PhotoImage(image1) 
        self.canvas1.create_image(0, 0, anchor=NW, image=self.image1_tk)

        image2 = Image.open(img_path2)  
        self.image2_tk = ImageTk.PhotoImage(image2) 
        self.canvas2.create_image(0, 0, anchor=NW, image=self.image2_tk)
        print('click!')

    
    def reset(self):
        self.inputs_list = []
        self.canvas.delete('inputs')
        self.text_box.delete('1.0', 'end')
    
    





def main():
    Visualization()


if __name__ == '__main__':
    main()