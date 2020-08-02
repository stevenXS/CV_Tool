import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import cv2
import ast
import os

csv_path = 'D:/dataset/global-wheat-detection/train.csv'
img_path = 'D:/dataset/global-wheat-detection/train'
save_path = 'D:/dataset/global-wheat-detection/train_clean.csv'




# 1、先可视化看看
'''
train = pd.read_csv(csv_path)
train[['x', 'y', 'w', 'h']] = pd.DataFrame(np.stack(train['bbox'].apply(lambda x: ast.literal_eval(x)))).astype(np.float32)
train['x1'] = train['x'] + train['w']
train['y1'] = train['y'] + train['h']
train['area'] = train['w'] * train['h']
# for i in range(95, 100, 1):
#     perc = np.percentile(train['area'], i)
#     print(f"{i} percentile of area is {perc}")
#
# for i in range(0, 5, 1):
#     perc = np.percentile(train['area'], i)
#     print(f"{i} percentile of area is {perc}")
#
# Train_Box = train[train['area']<100]  # clw note:  200, 300, 400 , 500 都找几张图看看，如果都看不清就全删
# Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])
# Train_Box = Train_Box.tail(4)
# Train_Box.head()
#
# grid_width = 2
# grid_height = 2
# images_id = ['6284044ed','ad256655b', '233cb8750', '6a8522f06']
# bbox_id = [36287, 40034, 114998, 119089]
# fig, axs = plt.subplots(grid_height, grid_width,
#                         figsize=(15, 15))
#
# for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):
#     ax = axs[int(i / grid_width), i % grid_width]
#     image = cv2.imread(f'../input/global-wheat-detection/train/{img_id}.jpg', cv2.IMREAD_COLOR)
#     box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]
#     cv2.rectangle(image,
#                   (box[0], box[1]),
#                   (box[2], box[3]),
#                   (220, 0, 0), 3)
#     ax.imshow(image.squeeze())
#
# plt.show()


Train_Box = train[train['area']>27456.23999999996]
Train_Box = Train_Box.sort_values(axis=0, ascending=True, by=['area'])
Train_Box = Train_Box.tail(15)
Train_Box.head(15)

grid_width = 3
grid_height = 5
images_id = ['b8ddb6c73', 'f1a8585e0', '51f2e0a05', '69fc3d3ff', '9adbfe503', '41c0123cc','a1321ca95', 'ad6e9eea2', '9a30dd802', 'd7a02151d', '409a8490c', '2cc75e9f5', 'a1321ca95', 'd067ac2b1', '42e6efaaa']
bbox_id = [128028, 53790, 53930, 1259, 54892, 173, 2169, 54702, 52868, 118211, 117344, 3687, 2159, 121633, 113947]
fig, axs = plt.subplots(grid_height, grid_width,
                        figsize=(15, 15))

for i, (img_id, box) in enumerate(zip(images_id, bbox_id)):
    ax = axs[int(i / grid_width), i % grid_width]
    image = cv2.imread(os.path.join(img_path, '{}.jpg'.format(img_id)) , cv2.IMREAD_COLOR)
    box = [int(Train_Box['x'][box]),int(Train_Box['y1'][box]),int(Train_Box['x1'][box]),int(Train_Box['y'][box])]
    cv2.rectangle(image,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 3)
    ax.set_title(img_id)
    ax.imshow(image.squeeze())

plt.show()
'''


# 2、Clean the bboxs to output new train
train = pd.read_csv(csv_path)
train[['x', 'y', 'w', 'h']] = pd.DataFrame(np.stack(train['bbox'].apply(lambda x: ast.literal_eval(x)))).astype(np.float32)
train['area'] = train['w'] * train['h']
print(train.shape)  # # 1477930


train_clean = train[train['area']>300]     # -> 147653
train_clean = train_clean[train['w']>10]     # -> 147620
train_clean = train_clean[train['h']>10]     # -> 147586
train_clean = train_clean.drop([173,2169,118211,52868,117344,3687,2159,121633,113947])  # 147577    默认删除这些行，因为axis默认是0；比如要删除列则需加上axis=1
# clw note: 前面3行的筛选并不会改变索引，而只是剔除不符合的索引，比如 0,1,2,3,4 剔除不符合的0,3之后，dataframe还剩下1,2,4，这样drop特定行就没问题了

print("remove {} boxes".format(train.shape[0] - train_clean.shape[0]))  # 216

train_clean.to_csv(save_path)