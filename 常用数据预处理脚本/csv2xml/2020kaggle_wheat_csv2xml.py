### csv info:
# image_id - the unique image ID
# width, height - the width and height of the images
# bbox - a bounding box, formatted as a Python-style list of [xmin, ymin, width, height]
# !! Not all images have bounding boxes.

################ 生成XML数据
from lxml.etree import Element, SubElement, tostring
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil

def save_xml(image_id, bbox, save_dir='./Annotations', width=2666, height=2000, channel=3):
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'kaggle_wheat'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_id + '.jpg'
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '%s' % width
    node_height = SubElement(node_size, 'height')
    node_height.text = '%s' % height
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '%s' % channel
    for xmin, ymin, w, h in bbox:
        left, top, right, bottom = xmin, ymin, xmin+w, ymin+h
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'wheat'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = '%s' % left
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = '%s' % top
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = '%s' % right
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = '%s' % bottom
    xml = tostring(node_root, pretty_print=True)
    save_xml = os.path.join(save_dir, image_id + '.xml')
    with open(save_xml, 'wb') as f:
        f.write(xml)
    return


def change2xml(label_dict={}):
    for image in label_dict.keys():
        image_name = os.path.split(image)[-1]
        bbox = label_dict.get(image, [])
        save_xml(image_name, bbox)
    return


if __name__ == '__main__':
    csv_path = 'D:/dataset/global-wheat-detection/train.csv'
    img_path = 'D:/dataset/global-wheat-detection/train'
    save_path = img_path  # save xml
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = pd.read_table(csv_path, sep=",")

    # 1、遍历csv，将box信息写入list，包括四个点 [ xmin，ymin，w，h ] 和 图片宽高
    img_box_info_dict = {}
    img_info_dict = {}
    for index, data in tqdm(df.iterrows()):
        image_id = data['image_id']
        box = data['bbox'][1:-1].split(', ')
        box = [float(i) for i in box]  # str需要先转为float，再转为int，否则会错 str '123.0' 不能转换为int类型
        box = np.array(box, dtype=np.int).reshape(1, 4)
        if image_id in img_box_info_dict.keys():
            img_box_info_dict[image_id] = np.concatenate((img_box_info_dict[image_id], box), axis=0)
        else:
            img_box_info_dict[image_id] = box
            img_info_dict[image_id] = [ int(data['width']),  int(data['height']) ]

    # 2、将刚才得到的信息，按image_id依次写入相应的xml内
    for image_id in tqdm(img_box_info_dict.keys()):
        img_w, img_h = img_info_dict[image_id][0], img_info_dict[image_id][0]
        box = img_box_info_dict[image_id]
        save_xml(image_id=image_id, bbox=box, save_dir=save_path, width=img_w, height=img_h, channel=3)

    # 3、如果有图片没有对应的xml文件，则自动生成一个空的xml，或者先挑出来作为负样本；
    negative_img_path = 'D:/dataset/global-wheat-detection/negative_sample'
    if not os.path.exists(negative_img_path):
        os.makedirs(negative_img_path)
    img_names = [name for name in os.listdir(img_path) if name.endswith('.jpg')]
    for img_name in tqdm(img_names):
        if not os.path.exists(os.path.join(save_path, img_name[:-4] + '.xml')):
            shutil.move(os.path.join(img_path, img_name), os.path.join(negative_img_path, img_name))

    print('----------------finish!------------------')