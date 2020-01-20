# written by hml
#coding:utf-8

import os
import sys
import shutil 
import numpy as np
import copy, cv2
import time
import xml.etree.ElementTree as ET

def process(xml_file_path, out_path, extract_classes):
    if not os.path.exists(xml_file_path): 
        print("skip '%s'"%(xml_file_path))
        return
    else:
        print("process '%s'"%(xml_file_path))

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
    
    objs = root.findall('object')

    ## none object left, return direct
    if len(objs) < 1:
        return
        
    ### clw note: none object left, write a bbox like(0, 0, 20, 20) for background.
    if len(objs) < 1:
        element = ET.Element('object')
        # 创建二级目录
        oneName = ET.Element('name')
        oneName.text = 'background'  # 二级目录的值 #结果展示：<id>1</id>
        onePose = ET.Element('pose')
        onePose.text = 'Unspecified'
        oneTruncated = ET.Element('truncated')
        oneTruncated.text = '1'
        oneDifficult = ET.Element('difficult')
        oneDifficult.text = '0'
        oneBndbox = ET.Element('bndbox')
        xmin = ET.Element('xmin')
        ymin = ET.Element('ymin')
        xmax = ET.Element('xmax')
        ymax = ET.Element('ymax')

        xmin.text = str(0)
        ymin.text = str(0)
        xmax.text = str(10)
        ymax.text = str(10)

        oneBndbox.append(xmin)
        oneBndbox.append(ymin)
        oneBndbox.append(xmax)
        oneBndbox.append(ymax)

        element.append(oneName)
        element.append(onePose)
        element.append(oneTruncated)
        element.append(oneDifficult)
        element.append(oneBndbox)
        root.append(element)

    ml_file_path = xml_file_path.replace('\\', '/') # clw added: for windows
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
    xml_root = "./train/"
    out_path = "./train_mini/"

    extract_classes = ["car"]  # clw note：即所有xml只保留car这个类别，删除其他类别；并且复制到另一个文件夹out_path下
    
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