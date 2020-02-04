#coding:utf-8
# written by clw: 配合classify_picture_into_dirs.py使用

import os
import sys
import shutil 
import numpy as np
import copy, cv2
import time
import xml.etree.ElementTree as ET


'''
功能说明：
扫描标签文件，只提取标签数组中给出的标签，提取完的结果默认在当前目录： ./train_extract/

标签数组示例:
["car", "people", "boat"]

使用示例：
python3.6 xml_extract_cls.py  [src_path]  [out_path]

'''


def process(xml_file_path, out_path, extract_classes):
    if not os.path.exists(xml_file_path): 
        print("skip '%s'"%(xml_file_path))
        return
    #else:
    #    print("process '%s'"%(xml_file_path))

    utf8_parser = ET.XMLParser(encoding='utf-8')
    tree = ET.parse(xml_file_path, parser=utf8_parser)
    root = tree.getroot()

    clear_objs = []

    ## clear object which not in 'extract_classes'
    for anno_id, obj in enumerate(root.iter('object')):
        name = obj.find('name').text
        if name not in extract_classes:
            clear_objs.append(obj)
    for obj in clear_objs:
        root.remove(obj)
    
    #objs = root.findall('object')
    ## none object left, return direct
    # if len(objs) < 1:
    #     return
    
    tree.write(os.path.join(out_path, xml_file_path.split("/")[-1]), encoding="utf-8")

    jpg_path = xml_file_path[:-3] + "jpg"
    png_path = xml_file_path[:-3] + "png"

    if os.path.exists(jpg_path):
        shutil.copy(jpg_path, out_path)
    elif os.path.exists(png_path):
        shutil.copy(png_path, out_path)
    pass


def extract_objects(xml_root, out_path, extract_classes):
    xml_file_names = []
    xml_file_paths = []
    xml_format = ".xml"
    for pos,_,fs in os.walk( xml_root ):
        for xml_file_name in fs:
            if xml_file_name.endswith(xml_format):
                xml_file_names.append(xml_file_name)
                xml_file_paths.append(os.path.join(pos, xml_file_name)) 
    for idx in range(len(xml_file_names)):
        process(xml_file_paths[idx], out_path, extract_classes)
    pass

if __name__ == "__main__":
    root = "/media/clwclw/data/2019xxxxxx/train_classify/"

    classes_folders = os.listdir(root)  # such as ["car", "people", "boat"]
    for class_folder in classes_folders:
        out_path = './extract_cls_output/' + class_folder
        extract_classes = class_folder
        xml_root = os.path.join(root, class_folder)


        if len(sys.argv) > 1:
            xml_root = sys.argv[1]

        if len(sys.argv) > 2:
            out_path = sys.argv[2]

        if xml_root is None or not os.path.exists(xml_root):
            print("not exists '%s'" %(xml_root))
            sys.exit(0)
        else:
            print("process '%s'" %(xml_root))

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        extract_objects(xml_root, out_path, extract_classes)
        pass