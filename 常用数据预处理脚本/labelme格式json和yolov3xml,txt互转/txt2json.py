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

__version__ = "4.5.6"

XML_EXT = '.xml'
ENCODE_METHOD = 'utf8'

cls_indexs = {0:"car", 1:"car", 2:'truck', 3:'suv'}

import base64
import PIL.Image
from io import BytesIO

## reference
## https://blog.csdn.net/haveanybody/article/details/86494063

##PIL转base64 
def pil_to_base64(img_pil):
    byte_buf = BytesIO()
    img_pil.save(byte_buf, format='JPEG')
    byte_data = byte_buf.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str
 
##base64转PIL
def base64_to_pil(base64_str):
    img_data = base64.b64decode(base64_str)
    byte_buf = BytesIO(img_data)
    img_pil = PIL.Image.open(byte_buf)
    return img_pil

##pil转cv2
def pil_to_cv2img(img_pil):
    img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    return img

##cv2转base64
def cv2_base64(img_cv2):
    base64_str = cv2.imencode('.jpg',img_cv2)[1].tostring()
    base64_str = base64.b64encode(base64_str)
    return base64_str 
  
##base64转cv2
def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString, np.uint8)  
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_cv2

'''
输入：png 或 jpg 之类图片文件
返回： 图片的base64
'''
def file_to_imgb64(image_path):
    imgb64 = None
    with open(image_path, "rb") as f:
        img_data = f.read()
        imgb64 = base64.b64encode(img_data).decode("utf-8")
    return imgb64

'''
输入：labelme 生成的 json文件
返回： cv2 图片
'''
def json_to_cv2img(json_file):
    data = json.load(open(json_file))
    imgb64 = data.get("imageData")
    img = base64_cv2(imgb64)
    return img


def txt2json(path_txt, json_path, img_path, img_h, img_w):
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
            bbox = (cx - label_w/2 , cy - label_h / 2, cx + label_w/2, cy + label_h/2, label_name)
            bbox_list.append(bbox)
    generate_json(json_path, img_path, img_h, img_w, bbox_list)
    pass

'''
bbox: x1, y1, x2, y2, label_name
'''
def generate_json(json_path, img_path, img_h, img_w, bbox_list=[]):
    img_name = os.path.split(img_path)[-1]
    # folder_name = os.path.dirname(json_path).split('/')[-1]

    shapes = []
    for bbox in bbox_list:
        x1, y1, x2, y2, label_name = bbox

        points = [[x1, y1],[x2, y2]]
        label_shape = dict(
            label=label_name,
            line_color=None,
            fill_color=None,
            points=points,
            shape_type="rectangle")
        shapes.append(label_shape)

    data = dict(
        version=__version__,
        flags={},
        shapes=shapes,
        imagePath=img_name,
        imageData=file_to_imgb64(img_path),
        imageHeight=img_h,
        imageWidth=img_w)

    with open(json_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
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
                json_path = txtf_path[:-4] + ".json"

                cv_img = None
                if os.path.exists(jpg_path):
                    img_path = jpg_path
                    cv_img = cv2.imread(jpg_path)
                elif os.path.exists(png_path):
                    img_path = png_path
                    cv_img = cv2.imread(png_path)
                if cv_img is not None:
                    h, w, _ = cv_img.shape
                    txt2json(txtf_path, json_path, img_path, h, w)
                else:
                    print("miss image of ", txtf_path)
    pass


if __name__ == "__main__":
    txt_path = "car/"

    if len(sys.argv) > 1:
        txt_path = sys.argv[1]

    main(txt_path)
    pass
