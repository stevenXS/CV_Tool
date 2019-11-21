import os
import sys
import xml.etree.ElementTree as ET
#sys.path.insert(0, '.')
import shutil

#### clw note: VOC2007 and 2012
ROOT = '/media/clwclw/data/VOCdevkit'   # Root folder where the VOCdevkit is located
TRAINSET = [
    ('2012', 'train'),
    ('2012', 'val'),
    ('2007', 'train'),
    ('2007', 'val'),
    ]

TESTSET = [('2007', 'test')]

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