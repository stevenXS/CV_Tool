# written by clw

import xml.etree.ElementTree as ET
import os

#xml_path = '/home/user/data/coarse_crease/train/Annotations/'  # need the final /
dataset_type = 'train'
xml_path = 'C:/Users/Administrator/Desktop/train_crop/' + dataset_type + '/Annotations/'

file_names = os.listdir(xml_path)
xml_file_names = []
for file_name in file_names:
    if file_name.endswith('.xml'):
        xml_file_names.append(file_name)

def modify_the_item_in_all_xmls(xml_path):
    for idx, xml_file_name in enumerate(xml_file_names):
        print('clw: idx = ', idx)
        #xml_file_path = os.path.join(xml_path, xml_file_name)
        xml_file_path = xml_path + xml_file_name
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        #size = root.find('size')
        #width = float(size.find('width').text)
        #height = float(size.find('height').text)

        '''
        # 修改xml的某个item值，比如path 或者  
        # path = root.find('path')
        # print('clw: path = ', path.text)
        # 判断格式
        format = ''
        if path.text.endswith('.jpg'):
            format = '.jpg'
        elif path.text.endswith('.png'):
            format = '.png'
        path.text = xml_file_path.split('.')[0] + format
        print('clw: modified_path = ', path.text)
        '''

        folder = root.find('folder')
        folder.text = dataset_type
        tree.write(xml_file_path)

modify_the_item_in_all_xmls(xml_path)