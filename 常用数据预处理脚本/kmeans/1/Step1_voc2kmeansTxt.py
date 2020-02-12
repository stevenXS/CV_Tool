#coding:utf-8

import sys
import os
import xml.etree.ElementTree as ET
from os import getcwd
from kmeans_lib import kmeans_anchors

#classes = ["car", "bus"]  # clw note:原来程序是把所有xml读进来,然后只对classes里面有的类别做聚类;其他忽略;
                           # 现在去掉这个限制,然后convert_annotation里面做了相应修改,屏蔽了三行代码
                           # 最后改成了kmeans聚类100次然后只输出acc的最大值和对应的anchor,和之前有所不同


def convert_annotation(xml_path):
    in_file = open(xml_path)
    tree=ET.parse(in_file)
    root = tree.getroot()

    annotations = ""
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        #if cls not in classes or int(difficult)==1:  # clw note:这里只对classes里面有的类进行聚类, 也可以把这三句屏蔽掉
        #    continue
        #cls_id = classes.index(cls)

        xmlbox = obj.find('bndbox')
        if float(xmlbox.find('xmax').text) > 1024 or float(xmlbox.find('ymax').text) > 1024:
            print('1111')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        #annotations += " " + ",".join([str(a) for a in b]) + ',' + str(cls_id)
        annotations += " " + ",".join([str(a) for a in b]) + ',' + str(cls)
    return annotations

def scan_annotations(img_path, save_path = "train.txt"):
    image_names = [i for i in  os.listdir(img_path) if i.endswith(".png") or i.endswith(".jpg") ]
    list_file = open(save_path, 'w')
    for image_name in image_names:
        xml_path = os.path.join(img_path, image_name[:-4] + '.xml') 
        content = os.path.join(img_path, image_name) + convert_annotation(xml_path) + '\n'
        list_file.write(content)
    list_file.close()
    pass

if __name__ == "__main__":
    #img_path = "/media/clwclw/data/textile/train/"
    img_path = "/media/clwclw/data/2019jiangyin/train_classes/hole"
    save_path = "train.txt"

    if len(sys.argv) > 1:
        img_path = sys.argv[1]

    if len(sys.argv) > 2:
        save_path = sys.argv[2]

    if not os.path.exists(img_path):
        print("not exists '%s'" %(img_path))
        sys.exit(0)

    scan_annotations(img_path, save_path)
    kmeans_anchors(save_path, 1)
    pass

