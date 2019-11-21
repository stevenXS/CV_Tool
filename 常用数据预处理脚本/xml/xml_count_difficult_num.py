# written by clw

import xml.etree.ElementTree as ET
import os
import glob

xml_path = '/media/clwclw/data/VOCdevkit/VOC2007_2012_clw/val'

xml_file_paths = glob.glob(os.path.join(xml_path, '*.xml'))


def count_difficult_nums(xml_file_paths):
    gt_nums_count = 0
    difficult_nums_count = 0
    for idx, xml_file_path in enumerate(xml_file_paths):
        print('xml counts: ', idx+1)
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for anno_id, obj in enumerate(root.iter('object')):
            gt_nums_count += 1
            difficult = obj.find('difficult').text
            if difficult != '0':
                print('difficult file: ', xml_file_path)
                difficult_nums_count += 1
    print('gt nums in all class:', gt_nums_count)
    print('difficult nums in all class:', difficult_nums_count)

count_difficult_nums(xml_file_paths)
