# written by clw

import xml.etree.ElementTree as ET
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil  # for move files to val folder

sns.set(color_codes=True)  # 暗蓝色背景，带格子

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

xmls_path = 'F:/crop'
img_format = '.png'

zh_dict = {'other': '其他', 'background': '背景', 'crease': '折痕'}

'''
函数功能：
输入:某一含有很多xml文件的路径，以及所有xml文件的列表
输出:每一类缺陷数量的统计结果，以dict的形式返回
'''
def count_gt_nums(xmls_path, xml_file_names):
    gt_nums_count = {}
    for idx, xml_file_name in enumerate(xml_file_names):
        #print('xml counts: ', idx)
        xml_file_path = os.path.join(xmls_path, xml_file_name)

        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for anno_id, obj in enumerate(root.iter('object')):
            name = obj.find('name').text
            if name not in zh_dict:
                print('error: key %s in xml %s not exist in zh_dict' % (name, xml_file_path))

            if zh_dict[name] in gt_nums_count:
                gt_nums_count[zh_dict[name]] += 1
            else:
                gt_nums_count[zh_dict[name]] = 1

    return gt_nums_count


# 1、统计所有xml，即整个数据集拥有的每个类别的缺陷数量
file_names = os.listdir(xmls_path)
xml_file_names = []
for file_name in file_names:
    if file_name.endswith('.xml'):
        xml_file_names.append(file_name)
gt_nums_count_all = count_gt_nums(xmls_path, xml_file_names)
print('gt nums:', gt_nums_count_all)


# 2、随机按一定比例（默认8:2，即验证集占20%）来划分训练集和验证集
#    比如数据集一共有100张图片，随机抽20张作为验证集
#    对验证集的类别数量进行统计，如果不能保证每一类缺陷的个数都在20%左右的一定范围，
#    则回到该步骤开始的地方，重新划分，重新统计，如此往复...
import random

xml_file_nums = len(xml_file_names)  # 统计xml总共的个数
val_split_ratio = 0.2
class_split_ratio_min = 0.16  # 每一类在验证集的object个数不能少于0.15，
class_split_ratio_max = 0.25  # 每一类在验证集的object个数不能多于0.25
count_split_times = 0  # 统计切分验证集的随机次数

while(1):
    bsuccessed_split = True
    random.shuffle(xml_file_names)
    xml_file_names_val = xml_file_names[:int(val_split_ratio * xml_file_nums)]
    gt_nums_count_val = count_gt_nums(xmls_path, xml_file_names_val)
    for key in gt_nums_count_all:
        if gt_nums_count_all[key] <= 10:  # 当然对于比如某个类总共只有10个样本，就不考虑验证集的样本数量了，因为样本数量太少了，根本不用关心这个类的训练结果；
            continue

        if key not in gt_nums_count_val:  # 如果验证集里面都没有这个key，说明不行，直接重新分验证集
            bsuccessed_split = False
            break

        # 如果某个类别满足条件，则继续下一个类别；否则直接重新分验证集；直到所有类别都满足条件，才可以切分
        if gt_nums_count_val[key] < gt_nums_count_all[key] * class_split_ratio_max and gt_nums_count_val[key] > gt_nums_count_all[key] * class_split_ratio_min:
            continue
        else:
            bsuccessed_split = False
            count_split_times += 1
            print('切分验证集经过的随机碰撞次数：', count_split_times)
            break

    if bsuccessed_split == False:  # 如果经过上面的切分，失败了，那么重新回到while循环开始
        continue
    else:                          # 满足切分条件
        print('切分验证集满足每个类别缺陷个数在%f~%f的范围要求，切分验证集经过的随机碰撞次数：%d' % (class_split_ratio_min, class_split_ratio_max, count_split_times))
        print('数据集xml数量:', xml_file_nums)
        print('验证集xml数量：', int(val_split_ratio * xml_file_nums))
        print('gt nums in train+val dataset:', gt_nums_count_all)
        print('gt nums in val dataset:', gt_nums_count_val)

        # 把xml_file_names_val里面的xml和对应的png移动到另一个文件夹，命名为val
        xmls_path_val = os.path.join(xmls_path, 'val')
        if not os.path.exists(xmls_path_val):
            os.mkdir(xmls_path_val)

        for xml_file_name in xml_file_names_val:
            shutil.move(os.path.join(xmls_path, xml_file_name),  os.path.join(xmls_path_val, xml_file_name))  # 移动xml
            img_file_name = xml_file_name.split('.')[0] + img_format
            shutil.move(os.path.join(xmls_path, img_file_name),  os.path.join(xmls_path_val, img_file_name))  # 移动img
        break

