'''
@Author: Conghao Wong
@Date: 1970-01-01 08:00:00
LastEditors: Conghao Wong
LastEditTime: 2020-08-23 00:35:18
@Description: file content
'''
import numpy as np
import os

model_name = 'ALL200NoReverseHalfSample'

class Result():
    def __init__(self, test_set):
        self.result_path = './results/result-{}{}.txt'.format(model_name, test_set)
        self.path_path = './results/path-{}{}.txt'.format(model_name, test_set)
        self.result = np.loadtxt(self.result_path)

        self.ade = self.result[0]
        self.fde = self.result[1]
        
        with open(self.path_path, 'r') as f:
            self.path = f.readline()


print(model_name+'&')
all_data = []
ade = []
fde = []
for dataset in [0, 1, 2, 3, 4]:
    r = Result(dataset)
    all_data.append(
        str(dataset) + ',' +
        r.path + ',' +
        str(r.ade) + ',' +
        str(r.fde) + '\n'
    )
    ade.append(r.ade)
    fde.append(r.fde)

    print('{:.2f}/{:.2f} & '.format(r.ade, r.fde), end='')

print('{:.2f}/{:.2f} \\\\'.format(np.mean(np.stack(ade)), np.mean(np.stack(fde))))

with open('{}.csv'.format(model_name), 'w+') as f:
    f.writelines(all_data)
    
    