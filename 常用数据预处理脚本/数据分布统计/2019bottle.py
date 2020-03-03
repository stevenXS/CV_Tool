## 统计某一类型的bbox的宽高分布  核密度图
import seaborn as sns
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)


ann_json = 'C:/Users/62349/Downloads/chongqing1_round2_train_20200213/annotations.json'
images_file = 'C:/Users/62349/Downloads/chongqing1_round2_train_20200213/images/'

with open(ann_json) as f:
    ann=json.load(f)


def visualize(image_dir, annotation_file, file_name):
    '''
    Args:
        image_dir (str): image directory
        annotation_file (str): annotation (.json) file path
        file_name (str): target file name (.jpg)
    Returns:
        None
    Example:
        image_dir = "./images"
        annotation_file = "./annotations.json"
        file_name = 'img_0028580.jpg'
        visualize(image_dir, annotation_file, file_name)
    '''
    image_path = os.path.join( image_dir, file_name )
    assert os.path.exists( image_path ), "image path not exist."
    assert os.path.exists( annotation_file ), "annotation file path not exist"
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    with open(annotation_file) as f:
        data = json.load(f)
    image_id = None
    for i in data['images']:
        if i['file_name'] == file_name:
            image_id = i['id']
            break
    if not image_id:
        print("file name {} not found.".format(file_name))
    large_img = True if max( image.shape[0], image.shape[1] ) > 1000 else False
    linewidth = 10 if large_img else 2
    for a in data['annotations']:
        if a['image_id'] == image_id:
            bbox = [int(b) for b in a['bbox']]
            bbox[2] = bbox[2] + bbox[0] - 1
            bbox[3] = bbox[3] + bbox[1] - 1
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), linewidth )
    if large_img:
        plt.figure(figsize=(12,10))
    else:
        plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.show()
    return

### seaborn绘制bbox框高尺寸图
categorys_w = [[] for j in range(len(ann['categories']))]
categorys_h = [[] for j in range(len(ann['categories']))]
for a in ann['annotations']:
    if a['category_id'] != 0:
        categorys_w[a['category_id'] - 1].append(round(a['bbox'][2], 2))
        categorys_h[a['category_id'] - 1].append(round(a['bbox'][3], 2))

class_id = 13  # clw note: 这里修改需要看bbox宽和高分布的那个class的id
data_list = list(zip(categorys_w[class_id-1], categorys_h[class_id-1]))
data = np.array(data_list)

### 或者随机生成一些数据
# np.random.seed(sum(map(ord, "distributions")))
# mean, cov = [0, 1], [(1, .5), (.5, 1)]
# data = np.random.multivariate_normal(mean, cov, 200)

df = pd.DataFrame(data, columns=["x", "y"])

sns.set(color_codes=True)
g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="m", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)  #画背景网格线
g.set_axis_labels("$X$", "$Y$")
plt.show()


