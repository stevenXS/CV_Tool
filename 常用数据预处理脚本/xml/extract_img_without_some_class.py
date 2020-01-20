# coding:utf-8


import os
import sys
import shutil
import numpy as np
import copy, cv2
import time
import xml.etree.ElementTree as ET

'''
功能说明：
主要是提取一些voc2007的图片作为背景图,以减少mmdetection自然场景人头检测的FP
提取完的图片默认在当前目录： ./train_extract/

标签数组示例:
['people', 'car']

使用示例：
python3.6 xml_extract_cls.py  [src_path]  [out_path]

'''


def process(xml_file_path, out_path, extract_classes):
    if not os.path.exists(xml_file_path):
        print("skip '%s'" % (xml_file_path))
        return
    else:
        print("process '%s'" % (xml_file_path))

    utf8_parser = ET.XMLParser(encoding='utf-8')
    tree = ET.parse(xml_file_path, parser=utf8_parser)
    root = tree.getroot()

    objs = root.findall('object')
    ### none object left, return direct
    if len(objs) < 1:
        return


    tree.write(os.path.join(out_path, xml_file_path.split("/")[-1]), encoding="utf-8")

    # ## clear object which not in 'extract_classes'
    # for anno_id, obj in enumerate(root.iter('object')):
    #     name = obj.find('name').text
    #     if name not in extract_classes:
    #         return

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
    for pos, _, fs in os.walk(xml_root):
        for xml_file_name in fs:
            if xml_file_name.endswith(xml_format):
                xml_file_names.append(xml_file_name)
                xml_file_paths.append(os.path.join(pos, xml_file_name))
    for idx in range(len(xml_file_names)):
        process(xml_file_paths[idx], out_path, extract_classes)
    pass


if __name__ == "__main__":
    xml_root = "/media/clwclw/data/VOCdevkit/VOC2007_2012_clw/val_try/"
    out_path = "./train_extract/"

    extract_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    if len(sys.argv) > 1:
        xml_root = sys.argv[1]

    if len(sys.argv) > 2:
        out_path = sys.argv[2]

    if xml_root is None or not os.path.exists(xml_root):
        print("not exists '%s'" % (xml_root))
        sys.exit(0)
    else:
        print("process '%s'" % (xml_root))

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    extract_objects(xml_root, out_path, extract_classes)
    pass