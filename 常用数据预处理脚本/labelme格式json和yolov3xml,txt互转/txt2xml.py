#coding:utf-8

## reference
## https://blog.csdn.net/x779250919/article/details/103927525

import os
import sys
import json
import numpy as np
import shutil
import cv2
import re

import codecs
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
from lxml import etree


XML_EXT = '.xml'
ENCODE_METHOD = 'utf8'


cls_indexs = {0:"car", 1:"car", 2:'truck', 3:'suv'}

def txt2xml(path_txt, xml_path, img_path, img_h, img_w):
    print("process", path_txt)
    bbox_list=[]
    with open(path_txt,'r') as ftxt:
        for line in ftxt:
            fields = re.sub(' +', ' ', line.strip()).split(' ')
            if len(fields) != 5:
                continue
            label_name = cls_indexs[int(fields[0])]
            cx = float(fields[1]) * img_w
            cy = float(fields[2]) * img_h
            label_w = float(fields[3]) * img_w
            label_h = float(fields[4]) * img_h           
            bbox = (cx - label_w/2 , cy - label_h / 2, label_w, label_h, label_name)
            bbox_list.append(bbox)
    generate_xml(xml_path, img_path, img_h, img_w, bbox_list)
    pass

'''
bbox: x, y, w, h, label_name
'''
def generate_xml(xml_path, img_path, img_h, img_w, bbox_list=[]):
    
    img_name = os.path.split(img_path)[-1]
    folder_name = os.path.dirname(xml_path).split('/')[-1]

    root = Element('annotation')

    folder = SubElement(root, 'folder')
    folder.text = folder_name

    filename = SubElement(root, 'filename')
    filename.text = img_name

    localImgPath = SubElement(root, 'path')
    localImgPath.text = img_path

    source = SubElement(root, 'source')
    database = SubElement(source, 'database')
    database.text = "Unknown"

    size_part = SubElement(root, 'size')
    width = SubElement(size_part, 'width')
    height = SubElement(size_part, 'height')
    depth = SubElement(size_part, 'depth')
    width.text = str(img_w)
    height.text = str(img_h)        
    depth.text ="3"

    segmented = SubElement(root, 'segmented')
    segmented.text = '0'
    
    for bbox in bbox_list:
        x, y, w, h, label_name = bbox
        object_item = SubElement(root, 'object')
        name = SubElement(object_item, 'name')
        name.text = label_name

        pose = SubElement(object_item, 'pose')
        pose.text = "Unspecified"

        truncated = SubElement(object_item, 'truncated') 
        truncated.text = "0"

        difficult = SubElement(object_item, 'difficult')
        difficult.text = "0"

        bndbox = SubElement(object_item, 'bndbox')
        
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(x)
        
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(y)

        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(x+w)

        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(y+h)

    ## 保存后的xml不缩进及换行
    # tree = ET.ElementTree(root)
    # ET.dump(root)
    # tree.write(xml_path, encoding="utf-8")

    ## 保存后的xml会缩进及换行
    out_file = codecs.open(xml_path, 'w',encoding=ENCODE_METHOD)       
    rough_string = ET.tostring(root, 'utf8')
    root = etree.fromstring(rough_string)
    prettifyResult = etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())      
    out_file.write(prettifyResult.decode('utf8'))
    out_file.close()
    pass

def main(txt_path = 'train/'):
    if txt_path is not None and os.path.exists(txt_path):
        print("process '%s'" %(txt_path))
    else:
        print("not exists '%s'" %(txt_path))
        return

    abs_txt_path = os.path.abspath( txt_path )
    for pos,_,fs in os.walk( abs_txt_path ):
        for f in fs:
            fmt = f[-4:].lower()
            if fmt == '.txt':
                txtf_path = os.path.join(pos, f)
                jpg_path = txtf_path[:-4] + ".jpg"
                png_path = txtf_path[:-4] + ".png"
                xml_path = txtf_path[:-4] + ".xml"

                cv_img = None
                if os.path.exists(jpg_path):
                    img_path = jpg_path
                    cv_img = cv2.imread(jpg_path)
                elif os.path.exists(png_path):
                    img_path = png_path
                    cv_img = cv2.imread(png_path)
                if cv_img is not None:
                    h, w, _ = cv_img.shape
                    txt2xml(txtf_path, xml_path, img_path, h, w)
                else:
                    print("missed image of ", txtf_path)
    pass


if __name__ == "__main__":
    txt_path = "labels/"

    if len(sys.argv) > 1:
        txt_path = sys.argv[1]

    main(txt_path)
    pass
