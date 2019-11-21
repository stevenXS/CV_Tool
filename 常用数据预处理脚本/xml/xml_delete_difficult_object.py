# written by clw

import xml.etree.ElementTree as ET
import os
import glob

xml_path = '/media/clwclw/data/VOCdevkit/VOC2007_2012_clw/train'

xml_file_paths = glob.glob(os.path.join(xml_path, '*.xml'))


def delete_difficult_objects(xml_file_paths):
    difficult_nums_count = 0
    for idx, xml_file_path in enumerate(xml_file_paths):
        print('xml counts: ', idx+1)
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for anno_id, obj in enumerate(root.findall('object')):  # clw note:这里不能用root.iter('object'),因为要删除元素,迭代器会指向下一个的下一个元素,从而导致跳过了一个item
            difficult = obj.find('difficult').text
            if difficult != '0':
                #print('difficult file: ', xml_file_path)
                root.remove(obj)
                difficult_nums_count += 1
        tree.write(xml_file_path)
    print('difficult nums in all class:', difficult_nums_count)

delete_difficult_objects(xml_file_paths)
