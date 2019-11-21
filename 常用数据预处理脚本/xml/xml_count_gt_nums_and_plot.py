# written by clw

import xml.etree.ElementTree as ET
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)  # 暗蓝色背景，带格子

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

xml_path = 'F:/【纺织品缺陷检测】/label/crop_xml_all2'

zh_dict = {'other': '其他', 'background': '背景', 'crease': '折痕'}

file_names = os.listdir(xml_path)
xml_file_names = []
for file_name in file_names:
    if file_name.endswith('.xml'):
        xml_file_names.append(file_name)


gt_nums_count = {}
def count_gt_nums(xml_path):
    for idx, xml_file_name in enumerate(xml_file_names):
        print('xml counts: ', idx)
        xml_file_path = os.path.join(xml_path, xml_file_name)

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

count_gt_nums(xml_path)
print('gt nums in all class:', gt_nums_count)

'''
对上面结果可视化
'''
df = pd.DataFrame([gt_nums_count])  # 必须加[]
sns.barplot(data=df)  # 柱状图
plt.show()


'''
统计缺陷总数
'''
total = 0
for key in gt_nums_count:
    total += gt_nums_count[key]
print('total gt nums:', total)

