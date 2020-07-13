import seaborn as sns
import pandas as pd
import numpy as np
import os
import cv2
from collections import Counter

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)

root_path = 'D:/2020water/train/'
xml_path =  os.path.join(root_path, 'box')
img_path =  os.path.join(root_path, 'image')

import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


### 统计所有图片的宽度、高度的分布情况
# img_width_list  = []
# img_height_list = []
# img_list =  os.listdir(img_path)
# for img_file in img_list:
#     if img_file.endswith(('.jpg','.png')):
#         img=cv2.imread(os.path.join(img_path, img_file))
#         print('%s,  宽：%d,  高：%d' % (img_file,img.shape[1], img.shape[0]))
#         img_width_list.append(img.shape[1])
#         img_height_list.append(img.shape[0])
#
# result_width = Counter(img_width_list)     # 统计不同宽度的图片数量
# result_height = Counter(img_height_list)   # 统计不同高度的图片数量
#
# print('clw:图片宽度统计（宽度：个数）：',result_width)
# print('clw:图片高度统计（高度：个数）：',result_height)





### seaborn绘制bbox框高尺寸图
bbox_flag = True
if bbox_flag:
    xml_names = os.listdir(xml_path)
    categorys_w = {'holothurian':[], 'echinus':[], 'scallop':[], 'starfish':[]}
    categorys_h = {'holothurian':[], 'echinus':[], 'scallop':[], 'starfish':[]}
    xml_names = [xml_name for xml_name in xml_names if xml_name.endswith('.xml')]
    for xml_name in xml_names:
        print('xml_name:', xml_name)
        tree = ET.parse(os.path.join(xml_path, xml_name))
        root = tree.getroot()
        for anno_id, obj in enumerate(root.iter('object')):
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            label = obj.find('name').text
            w = xmax - xmin
            h = ymax - ymin
            categorys_w[label].append(w)
            categorys_h[label].append(h)


    #class_name = 'holothurian'
    class_name = 'starfish'
    data_list = list(zip(categorys_w[class_name], categorys_h[class_name]))
    print(sorted(data_list)[:11])

    ### 统计下该类别的高宽比分布
    # ratios = []
    # for wh in data_list:
    #     ratios.append(wh[1] / wh[0])  # 注意是高宽比
    # ratios = sorted(ratios)
    # print(ratios)




    # ### 准备seaborn作图
    # data = np.array(data_list)
    # # 或者随机生成一些数据
    # # np.random.seed(sum(map(ord, "distributions")))
    # # mean, cov = [0, 1], [(1, .5), (.5, 1)]
    # # data = np.random.multivariate_normal(mean, cov, 200)
    # df = pd.DataFrame(data, columns=["x", "y"])
    #
    # sns.set(color_codes=True)
    # g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="m")
    # g.plot_joint(plt.scatter, c="m", s=30, linewidth=1, marker="+")
    # g.ax_joint.collections[0].set_alpha(0)  #画背景网格线
    # g.set_axis_labels("$X$", "$Y$")
    # plt.show()


