#coding:utf-8

## reference
## https://blog.csdn.net/x779250919/article/details/103927525

import os
import sys
import json
import numpy as np
import shutil

import codecs
from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element, SubElement
from lxml import etree


XML_EXT = '.xml'
ENCODE_METHOD = 'utf8'


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


def json2xml(path_json, xml_path, img_path):
    print("process", path_json)
    bbox_list=[]
    with open(path_json,'r') as path_json:
        jsonx=json.load(path_json)
        img_w = jsonx["imageWidth"]
        img_h = jsonx["imageHeight"]
        for shape in jsonx['shapes']:
            points = np.array(shape['points'])
            x1, y1=points[0]
            x2, y2=points[1]
            label_w = abs(x2 - x1)
            label_h = abs(y2 - y1)
            bbox = (x1, y1, label_w, label_h, shape['label'])
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


def main(json_path = 'train/'):
    if json_path is not None and os.path.exists(json_path):
        print("process '%s'" %(json_path))
    else:
        print("not exists '%s'" %(json_path))
        return
    
    abs_json_path = os.path.abspath( json_path )
    for pos,_,fs in os.walk( abs_json_path ):
        for f in fs:
            fmt = f[-5:].lower()
            if fmt == '.json':
                jsonf_path = os.path.join(pos, f)
                jpg_path = jsonf_path[:-5] + ".jpg"
                png_path = jsonf_path[:-5] + ".png"
                xml_path = jsonf_path[:-5] + ".xml"

                img_path = None
                if os.path.exists(jpg_path):
                    img_path = jpg_path
                elif os.path.exists(png_path):
                    img_path = png_path
                if img_path is not None:
                    json2xml(jsonf_path, xml_path, img_path)
                else:
                    print("miss image of ", jsonf_path)
    pass


if __name__ == "__main__":
    json_path = "car/"

    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    main(json_path)
    pass
