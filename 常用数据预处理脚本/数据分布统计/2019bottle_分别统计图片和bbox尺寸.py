# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:50:25 2019
@author: clwclw

统计结果：
图片共计4516张，其中4105张瓶盖, 宽和高均为658x492
                    411张瓶身, 宽和高均为4096x3000

其中3218张图片只有1个object，1张图片有21个object，1张有16个object，其他都是10个以内
bbox共计：6945
bbox宽度：6.14~2497
bbox高度：3.85~2101
宽高比：0.05~25，最大25:1

每个缺陷类别数量统计：
0 --- 1619
1 --- 705
2 --- 656
3 --- 480
4 --- 614
5 --- 186
6 --- 384
7 --- 443
8 --- 489
9 --- 199
目前的数据只有10类缺陷，官方一共11类；



"""





#img_and_anno_root = '/mfs/home/fangyong/data/guangdong/round2/train/'
img_and_anno_root = 'C:/Users/62349/Downloads/chongqing1_round1_train1_20191223_split/'
#img_and_anno_root ='K:/deep_learning/dataset/2019tianchi/train/'
###########################################################################################

import os
from PIL import Image
from collections import Counter
img_path = img_and_anno_root + 'images/'

######## 1、统计所有图片的宽度、高度的分布情况

# clw note：这里图片用宽*高表示
img_width_list  = []
img_height_list = []
img_list =  os.listdir(img_path)
for img_file in img_list:
    if img_file.endswith(('.jpg','.png')):
        img=Image.open(img_path + '/' + img_file)
        ratio = img.size[1]/ img.size[0]
        #print('%s,  宽：%d,  高：%d'%(img_file,img.size[0], img.size[1]))
        img_width_list.append(img.size[0])
        img_height_list.append(img.size[1])

print("clw:所有图片最大的宽度为%d,对应的图片名为%s" %(max(img_width_list) , img_list[img_width_list.index(max(img_width_list))]  ))
print("clw:所有图片最大的高度为%d,对应的图片名为%s" %(max(img_height_list), img_list[img_height_list.index(max(img_height_list))]  ))

result_width = Counter(img_width_list)     # 统计不同宽度的图片数量
result_height = Counter(img_height_list)   # 统计不同高度的图片数量

print('clw:图片宽度统计（宽度：个数）：',result_width)
print('clw:图片高度统计（高度：个数）：',result_height)

###########################################################################################



#### 2、统计所有bbox的宽度、高度、宽高比最值和分布情况
import json

annFile = img_and_anno_root + 'annotations2.json'
#annFile = img_and_anno_root + 'Annotations/val_.json'
file = open(annFile, "rb")
datas = json.load(file)
#data_list = sorted(data_list,key = lambda e:e['name'],reverse = True)

image_ids = []
defect_names = []

ann_list = datas['annotations']

bbox_width = []
bbox_height = []
bbox_ratio = []
bbox_area = []
for data in ann_list:
    image_ids.append(data['image_id'])
    defect_names.append(data['category_id'])
    bbox_width.append(data['bbox'][2])  # clw note:json is x,y,w,h
    bbox_height.append(data['bbox'][3])
    bbox_ratio.append(data['bbox'][2] / data['bbox'][3])
    bbox_area.append(data['bbox'][2] * data['bbox'][3])

print('clw:bbox数量共计：', len(ann_list))
print('clw:image数量共计：', len(set(image_ids)))
print('clw:bbox宽度：', sorted(bbox_width))
print('clw:bbox高度：', sorted(bbox_height))
print('clw:bbox宽高比）：',sorted(bbox_ratio))
#print('clw:bbox面积）：',sorted(bbox_area))



#########################
### 作图
########################
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
#plt.style.use('ggplot') #使用'ggplot'风格美化显示的图表
# font = {'family':'SimHei'} #设置使用的字体（需要显示中文的时候使用）
# plt.rc('font',**font) #设置显示中文，与字体配合使用
import matplotlib as mpl #新增包
#from matplotlib.ticker import MultipleLocator, FormatStrFormatter #新增函数
mpl.rcParams['font.size'] = 12 #设置字体大小
plt.rcParams['font.family'] = 'SimSun'  # 设置全局的字体
custom_font = mpl.font_manager.FontProperties(fname='../数据标注可视化/coco数据可视化/simsun.ttf') #导入字体文件
#custom_font = mpl.font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc') #导入字体文件

# （1）一幅图的gt数量的统计直方图
bbox_counts = Counter(image_ids)
#########################################
bbox_count_list = [i[1] for i in bbox_counts.most_common()]  # 形如[37, 25, 24, 24, ...., 1, 1, 1, 1, 1]
bbox_count_max = max(bbox_count_list)

#plt.hist(bbox_count_list, bins=bbox_count_max, normed=False, log=True, color='cornflowerblue')  # clw note：bins指定总共有几条条状图； normed是否归一化；log取对数
# 可以用以下功能代替plt.hist()，就可以手动添加一些想要的功能，比如在bar上面把数字打出来
num_list = []
for i in range(bbox_count_max+1):
    num_list.append(bbox_count_list.count(i))
bbb = plt.bar(range(bbox_count_max+1), num_list, color='cornflowerblue', tick_label=range(bbox_count_max+1), log=True)  # x：bar的横坐标
for x, y in enumerate(num_list):
    if y == 0:  # 直方图如果高度为0，就不用在上面加个数字0了，否则很难看
        plt.text(x, y + 0.7, y, ha='center', va='bottom', fontsize=10)
    elif y <= 3:
        plt.text(x, y + 0.1, y, ha='center', va='bottom', fontsize=10)  # font='/home/user/clwclw/simsun.ttf'
    else:
        plt.text(x, y+1, y, ha='center', va='bottom', fontsize=10)  # font='/home/user/clwclw/simsun.ttf'   # 前三个参数：x,y:表示坐标值上的值，string:表示说明文字

plt.title('含有一定数量object的图片个数统计', fontsize=24, fontproperties=custom_font)
plt.xlabel('object个数', fontsize=14, fontproperties=custom_font)
plt.ylabel('图片数量', fontsize=14, fontproperties=custom_font)


# （2）34类缺陷，每一类缺陷个数的直方图
defect_counts = Counter(defect_names)
num_list = []
# name_list = ['\u7834\u6d1e', '\u6c34\u6e0d', '\u6cb9\u6e0d', '\u6c61\u6e0d',
#                  '\u4e09\u4e1d', '\u7ed3\u5934', '\u82b1\u677f\u8df3', '\u767e\u811a', '\u6bdb\u7c92',
#                  '\u7c97\u7ecf', '\u677e\u7ecf', '\u65ad\u7ecf', '\u540a\u7ecf', '\u7c97\u7ef4',
#                  '\u7eac\u7f29', '\u6d46\u6591', '\u6574\u7ecf\u7ed3', '\u661f\u8df3', '\u8df3\u82b1',
#                  '\u65ad\u6c28\u7eb6', '\u7a00\u5bc6\u6863', '\u6d6a\u7eb9\u6863', '\u8272\u5dee\u6863', '\u78e8\u75d5',
#                  '\u8f67\u75d5', '\u4fee\u75d5', '\u70e7\u6bdb\u75d5', '\u6b7b\u76b1', '\u4e91\u7ec7',
#                  '\u53cc\u7eac', '\u53cc\u7ecf', '\u8df3\u7eb1', '\u7b58\u8def', '\u7eac\u7eb1\u4e0d\u826f']
# final_list = [' \u7834\n \u6d1e', '\u6c34\n\u6e0d', '\u6cb9\n\u6e0d', '\u6c61\n\u6e0d',   # clw note：很奇怪，如果第一个元素没有空格，或者不去掉回车，那么后面元素都会在左右两侧有一段缺失，可以看下效果
#             '\u4e09\n\u4e1d', '\u7ed3\n\u5934', '\u82b1\n\u677f\n\u8df3', '\u767e\n\u811a', '\u6bdb\n\u7c92',
#             '\u7c97\n\u7ecf', '\u677e\n\u7ecf', '\u65ad\n\u7ecf', '\u540a\n\u7ecf', '\u7c97\n\u7ef4',
#             '\u7eac\n\u7f29', '\u6d46\n\u6591', '\u6574\n\u7ecf\n\u7ed3', '\u661f\n\u8df3', '\u8df3\n\u82b1',
#             '\u65ad\n\u6c28\n\u7eb6', '\u7a00\n\u5bc6\n\u6863', '\u6d6a\n\u7eb9\n\u6863', '\u8272\n\u5dee\n\u6863', '\u78e8\n\u75d5',
#             '\u8f67\n\u75d5', '\u4fee\n\u75d5', '\u70e7\n\u6bdb\n\u75d5', '\u6b7b\n\u76b1', '\u4e91\n\u7ec7',
#             '\u53cc\n\u7eac', '\u53cc\n\u7ecf', '\u8df3\n\u7eb1', '\u7b58\n\u8def', '\u7eac\n\u7eb1\n\u4e0d\n\u826f']

# clw note: chinese not support
name_list = ['沾污','错花','水印', '花毛', '缝头', '缝头印', '虫粘','破洞', '褶子', '织疵', '漏印', '蜡斑', '色差', '网折', '其他']
final_list = [' 沾污','错花','水印', '花毛', '缝头', '缝头印', '虫粘','破洞', '褶子', '织疵', '漏印', '蜡斑', '色差', '网折', '其他']


for index, name in enumerate(name_list):
    num_list.append(defect_counts[index+1])

fig, ax = plt.subplots()
bbb = ax.bar(range(len(name_list)), num_list, color='cornflowerblue', tick_label=final_list)  # x：bar的横坐标
#bbb = ax.barh(range(len(name_list)), num_list, color='cornflowerblue', tick_label=name_list)  # clw note：如果改成水平，后面ax.text内坐标也需要调整
for x, y in enumerate(num_list):
    ax.text(x, y+1, y, ha='center', va='bottom', fontsize=10)  # font='/home/user/clwclw/simsun.ttf'   # 前三个参数：x,y:表示坐标值上的值，string:表示说明文字

plt.title('每一类缺陷个数统计', fontsize=24, fontproperties=custom_font)
plt.xlabel('类别', fontsize=14, fontproperties=custom_font)
plt.ylabel('bounding box数量', fontsize=14, fontproperties=custom_font)
plt.show()
