# written by clw

import xml.etree.ElementTree as ET
import os

xml_path = 'D:/dataset/v3/'

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
        for obj in root.iter('object'):
            name = obj.find('name')
            if name.text == 'pilling':
                name.text = 'dead cotton'
            elif name.text == 'misclip':
                name.text = 'broken edge'
            elif name.text == 'needle gall':
                name.text = 'needle line'
            elif name.text == 'pulling out':
                name.text = 'broken yarn'
            elif name.text == 'dissection wrong':
                name.text = 'drop needle'
            elif name.text == 'slub':
                name.text = 'coarse'
        tree.write(xml_file_path)

modify_the_item_in_all_xmls(xml_path)