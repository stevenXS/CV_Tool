'''
# 功能：把原始目录结构的voc2007、voc2012等数据集，按照训练集和测试集划分好，
#       分别放置在train_path和test_path下

VOC2007数据集共包含：
训练集（train+val共有5011张，其中train.txt有2501张，val.txt有2510张），
测试集（4952幅），共计9963幅图，
（自注：对于VOC2012，train+val共有11540张，train.txt有5717张，val.txt有5823张；  加上VOC2007共计16551张；
另外对于yolov3一般是把2007和2012的train.txt和val.txt都放一块作为训练集，然后把比如2007的test.txt作为测试集，
可以用这个命令来合并txt：cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt）
'''

import os
import sys
import xml.etree.ElementTree as ET
#sys.path.insert(0, '.')
import shutil

#### clw note: VOC2007 and 2012
ROOT = 'E:/deep_learning/dataset/VOCdevkit/'   # Root folder where the VOCdevkit is located
TRAINSET = [
    ('2007', 'val'),
    ]

TESTSET = []

train_xml = []
train_img = []
for (year, img_set) in TRAINSET:
    with open(f'{ROOT}/VOC{year}/ImageSets/Main/{img_set}.txt', 'r') as f:
        ids = f.read().strip().split()
    train_xml += [f'{ROOT}/VOC{year}/Annotations/{xml_id}.xml' for xml_id in ids]
    train_img += [f'{ROOT}/VOC{year}/JPEGImages/{xml_id}.jpg' for xml_id in ids]
assert len(train_img) == len(train_xml)
print('train dataset xml_files_num:', len(train_xml))

test_xml = []
test_img = []
for (year, img_set) in TESTSET:
    with open(f'{ROOT}/VOC{year}/ImageSets/Main/{img_set}.txt', 'r') as f:
        ids = f.read().strip().split()
    test_xml += [f'{ROOT}/VOC{year}/Annotations/{xml_id}.xml' for xml_id in ids]
    test_img += [f'{ROOT}/VOC{year}/JPEGImages/{xml_id}.jpg' for xml_id in ids]
assert len(test_img) == len(test_xml)
print('test dataset xml_files_num:', len(test_xml))


train_path = './train'
if not os.path.exists(train_path):
    os.mkdir(train_path)
for idx, xml_file_path in enumerate(train_xml):
    shutil.copy(xml_file_path, train_path)
    print('train_xml_file_count:', idx + 1)
for idx, img_file_path in enumerate(train_img):
    shutil.copy(img_file_path, train_path)
    print('train_img_file_count:', idx + 1)

test_path = './val'
if not os.path.exists(test_path):
    os.mkdir(test_path)
for idx, xml_file_path in enumerate(test_xml):
    shutil.copy(xml_file_path, test_path)
    print('test_xml_file_count:', idx + 1)
for idx, img_file_path in enumerate(test_img):
    shutil.copy(img_file_path, test_path)
    print('test_img_file_count:', idx + 1)


print('finish!')