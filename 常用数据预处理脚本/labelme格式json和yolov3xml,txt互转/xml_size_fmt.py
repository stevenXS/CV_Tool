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
ENCODE_METHOD = 'utf-8'

IMG_FMT_SIZE = 1000
LABEL_MAX_SIZE = 30
LABEL_FMT_SIZE = 20
FMT_LABEL = True


def fmt_xml(xml_file_path, new_xml_path, img_path):
    if xml_file_path == new_xml_path:
        print("skip", xml_file_path, "->", new_xml_path)
        return

    # print("process", xml_file_path)

    img = cv2.imread(img_path)
    fmt_img = np.zeros((IMG_FMT_SIZE, IMG_FMT_SIZE, 3), np.uint8)
    fmt_img[:,:,0] = np.mean(img[:,:,0])        # B_mean
    fmt_img[:,:,1] = np.mean(img[:,:,1])        # G_mean
    fmt_img[:,:,2] = np.mean(img[:,:,2])        # R_mean

    img_h, img_w, _ = img.shape
    large_size = img_w if img_w>img_h else img_h
    scale = IMG_FMT_SIZE / large_size

    utf8_parser = ET.XMLParser(encoding=ENCODE_METHOD)
    tree = ET.parse(xml_file_path, parser=utf8_parser)
    root = tree.getroot()

    size_obj = root.find('size')
    size_obj.find("width").text = str(IMG_FMT_SIZE)
    size_obj.find("height").text = str(IMG_FMT_SIZE)
    
    objs = root.findall('object')

    label_size_list = []
    for obj in objs:
        bndbox_obj =obj.find('bndbox')
        xmin = bndbox_obj.find('xmin')
        ymin = bndbox_obj.find('ymin')
        xmax = bndbox_obj.find('xmax')
        ymax = bndbox_obj.find('ymax')
        x1 = float(xmin.text) * scale
        y1 = float(ymin.text) * scale
        x2 = float(xmax.text) * scale
        y2 = float(ymax.text) * scale
        xmin.text = str(x1 )
        ymin.text = str(y1)
        xmax.text = str(x2)
        ymax.text = str(y2)
        label_w = abs(x2-x1)
        label_h = abs(y2-y1)
        label_size = label_w if label_w>label_h else label_h
        label_size_list.append(label_size)
    
    min_size = np.min(label_size_list)
    if FMT_LABEL and min_size > LABEL_MAX_SIZE:
        label_scale = LABEL_FMT_SIZE / min_size
        scale = scale * label_scale
        for obj in objs:
            bndbox_obj =obj.find('bndbox')
            xmin = bndbox_obj.find('xmin')
            ymin = bndbox_obj.find('ymin')
            xmax = bndbox_obj.find('xmax')
            ymax = bndbox_obj.find('ymax')
            x1 = float(xmin.text) * label_scale
            y1 = float(ymin.text) * label_scale
            x2 = float(xmax.text) * label_scale
            y2 = float(ymax.text) * label_scale
            xmin.text = str(x1 )
            ymin.text = str(y1)
            xmax.text = str(x2)
            ymax.text = str(y2)

    scale_w = int(img_w*scale)
    scale_h = int(img_h*scale)
    img_scale = cv2.resize(img, (scale_w, scale_h))
    fmt_img[:scale_h, :scale_w] = img_scale

    new_img_path = new_xml_path[:-3] + "jpg"
    # cv2.imwrite(new_img_path, fmt_img)

    gray_img=cv2.cvtColor(fmt_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(new_img_path, gray_img)

    tree.write(new_xml_path, encoding=ENCODE_METHOD)
    pass


def main(xml_path = 'train/', out_path="output/"):
    if xml_path is not None and os.path.exists(xml_path):
        print("process '%s'" %(xml_path))
    else:
        print("not exists '%s'" %(xml_path))
        return

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    cnt = 0
    abs_out_path = os.path.abspath( out_path )
    abs_xml_path = os.path.abspath( xml_path )
    for pos,_,fs in os.walk( abs_xml_path ):
        for f in fs:
            fmt = f[-4:].lower()
            if fmt == XML_EXT:
                xmlf_path = os.path.join(pos, f)
                jpg_path = xmlf_path[:-4] + ".jpg"
                png_path = xmlf_path[:-4] + ".png"

                img_path = None
                if os.path.exists(jpg_path):
                    img_path = jpg_path
                elif os.path.exists(png_path):
                    img_path = png_path
                if img_path is not None:
                    new_xml_path = os.path.join(abs_out_path, f)
                    fmt_xml(xmlf_path, new_xml_path, img_path)
                    cnt += 1
                else:
                    print("miss image of ", xmlf_path)
    print(cnt, "done!")
    pass


if __name__ == "__main__":
    xml_path = "labels/"
    out_path = "./output"

    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    
    if len(sys.argv) > 3:
        if sys.argv[3] == "True" or sys.argv[3] == "true":
            FMT_LABEL = True
        else:
            FMT_LABEL = False
    main(xml_path, out_path)
    pass
