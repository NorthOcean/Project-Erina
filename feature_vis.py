'''
@Author: Conghao Wong
@Date: 2020-07-15 17:05:12
@LastEditors: Conghao Wong
@LastEditTime: 2020-07-15 23:06:32
@Description: file content
'''


import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def cal_pca(data, dim=2):
    pca = PCA(n_components=2)
    return pca.fit_transform(data)

def cal_tsne(data, dim=2):
    tsne = TSNE(n_components=2)
    return tsne.fit_transform(data)


feature_file = './features.npy'
traj_file = './trajs.npy'

features = np.load(feature_file)
trajs = np.load(traj_file)

features_pca = cal_tsne(features.reshape([-1, 128*8]))
f_min = np.min(features_pca, axis=0)
f_max = np.max(features_pca, axis=0)

f_regular = (features_pca - f_min)/(f_max-f_min)

# plt.figure(figsize=[32, 18])
# for i, data in enumerate(features_pca):
#     plt.plot(data[0], data[1], 'ro')
#     plt.text(data[0], data[1], '{}'.format(i), weight="bold", color="b")

fig = plt.figure(figsize=[150, 150])
for i, (data, traj1) in tqdm(enumerate(zip(f_regular, trajs))):
    ax2 = fig.add_axes([data[0], data[1], 0.01, 0.01])
    ax2.set_title('{}'.format(i))  
    ax2.plot(traj1.T[0][:8], traj1.T[1][:8], '-ro')
    ax2.plot(traj1.T[0][8:], traj1.T[1][8:], '-go')
    # ax2.set_xlim(-10, 30)
    # ax2.set_ylim(-10, 30)
    ax2.axis('scaled')

plt.savefig('./feature_vis/feafull.png')
plt.close()
# np.savetxt('./ff.txt', features_pca)

# for index, (feature1, traj1) in enumerate(tqdm(zip(features, trajs))):
#     f1_seq = np.abs(feature1[:, :64])
#     f1_state = np.abs(feature1[:, 64:])

#     f1_seq_m = np.mean(f1_seq, axis=1).reshape([-1])
#     f1_state_m = np.mean(f1_state, axis=1).reshape([-1])

#     plt.figure()

#     plt.subplot(1, 3, 1)
#     plt.plot(f1_seq_m, '-ro')
#     plt.ylim(0, f1_seq_m.max() * 1.1)


#     plt.subplot(1, 3, 2)
#     plt.plot(f1_state_m, '-bo')
#     plt.ylim(0, f1_state_m.max() * 1.1)


#     plt.subplot(1, 3, 3)
#     plt.plot(traj1.T[0][:8], traj1.T[1][:8], '-r*')
#     plt.plot(traj1.T[0][8:], traj1.T[1][8:], '-go')
#     plt.axis('scaled')
#     plt.savefig('./feature_vis/traj{}.png'.format(index))
#     plt.close()

# np.savetxt('./f1.txt', )
# np.savetxt('./f2.txt', )
print('!')

