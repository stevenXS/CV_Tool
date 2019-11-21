### 自注：腾讯优图项目ObjectDetection-OneStageDet的一个脚本，修改之后可以针对自己的数据集生成pkl文件

#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Transform annotations for VOCdevkit to the brambox pickle format
#

# modified by mileistone

import os
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, '.')
import brambox.boxes as bbb

DEBUG = True        # Enable some debug prints with extra information
#ROOT = '/mfs/home/xy/data/VOCdevkit'       # Root folder where the VOCdevkit is located
ROOT = '/media/clwclw/data/2019tianchi/VOC'

# #### clw note: VOC2007 and 2012
# ROOT = '/media/clwclw/data/VOCdevkit'
# TRAINSET = [
#     ('2012', 'train'),
#     ('2012', 'val'),
#     ('2007', 'train'),
#     ('2007', 'val'),
#     ]
#
# TESTSET = [
#     ('2007', 'test'),
#     ]

def identify(xml_file):
    root_dir = ROOT
    root = ET.parse(xml_file).getroot()
    folder = root.find('folder').text  # clw note: 如VOC
    filename = root.find('filename').text 
    #return f'{root_dir}/{folder}/JPEGImages/{filename}'  # clw note: for VOC2007 and VOC2012
    return f'{root_dir}/{folder}/images/{filename}'  # clw modify


if __name__ == '__main__':
    print('Getting training annotation filenames')

    # train = []
    # for (year, img_set) in TRAINSET:
    #     with open(f'{ROOT}/VOC{year}/ImageSets/Main/{img_set}.txt', 'r') as f:
    #         ids = f.read().strip().split()
    #     train += [f'{ROOT}/VOC{year}/Annotations/{xml_id}.xml' for xml_id in ids]


    ### clw note:自定义数据集
    train = os.listdir(ROOT + '/train/Annotations')
    train = [ROOT + '/train/Annotations/' + item for item in train]

    if DEBUG:
        print(f'\t{len(train)} xml files')

    print('Parsing training annotation files')
    train_annos = bbb.parse('anno_pascalvoc', train, identify)
    # Remove difficult for training
    for k,annos in train_annos.items():
        for i in range(len(annos)-1, -1, -1):
            if annos[i].difficult:
                del annos[i]

    print('Generating training annotation file')
    bbb.generate('anno_pickle', train_annos, f'{ROOT}/onedet_cache/train.pkl')


    print('Getting testing annotation filenames')

    # test = []
    # for (year, img_set) in TESTSET:
    #     with open(f'{ROOT}/VOC{year}/ImageSets/Main/{img_set}.txt', 'r') as f:
    #         ids = f.read().strip().split()
    #     test += [f'{ROOT}/VOC{year}/Annotations/{xml_id}.xml' for xml_id in ids]

    ### clw note:自定义数据集
    test = os.listdir(ROOT + '/val/Annotations')
    test = [ROOT + '/val/Annotations/' + item for item in test]

    if DEBUG:
        print(f'\t{len(test)} xml files')

    print('Parsing testing annotation files')
    test_annos = bbb.parse('anno_pascalvoc', test, identify)

    print('Generating testing annotation file')
    bbb.generate('anno_pickle', test_annos, f'{ROOT}/onedet_cache/test.pkl')

