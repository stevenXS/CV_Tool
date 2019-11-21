# written by clw
# 功能：之前用labelImg标注一些破洞的时候，太贴合了，留出的边缘不够，因此考虑写个脚本来扩边
# 比如之前xmin,ymin,xmax,ymax分别为100,100,200,200，扩10个pixel变为90,90,210,210，超出边界则退回到边界

import xml.etree.ElementTree as ET
import os

#xml_path = 'D:/dataset/try/'
xml_path = 'C:/Users/62349/Desktop/v3/'
img_width = 1024
img_height = 1024

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
            if name.text in ['hole', 'dusty']:
                bbox = obj.find('bndbox')

                edge_distance = 20
                xmin = int(bbox.find('xmin').text)
                if xmin - edge_distance > 0:
                    bbox.find('xmin').text = str(xmin - edge_distance)
                else:
                    bbox.find('xmin').text = '0'

                ymin = int(bbox.find('ymin').text)
                if  ymin - edge_distance > 0:
                    bbox.find('ymin').text = str(ymin - edge_distance)
                else:
                    bbox.find('ymin').text = '0'

                xmax = int(bbox.find('xmax').text)
                if xmax + edge_distance < img_width:
                    bbox.find('xmax').text = str(xmax + edge_distance)
                else:
                    bbox.find('xmax').text = str(img_width)

                ymax = int(bbox.find('ymax').text)
                if ymax + edge_distance < img_height:
                    bbox.find('ymax').text = str(ymax + edge_distance)
                else:
                    bbox.find('ymax').text = str(img_height)

        tree.write(xml_file_path)

modify_the_item_in_all_xmls(xml_path)
print('end!')