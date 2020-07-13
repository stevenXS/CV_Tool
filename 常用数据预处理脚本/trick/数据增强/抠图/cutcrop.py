#!/usr/bin/env python
# coding=UTF-8





from pycocotools.coco import COCO
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

count=0
img_and_anno_root = 'C:/Users/62349/Downloads/chongqing1_round2_train_20200213/'
img_path = img_and_anno_root + 'pingshen/'
#annFile = img_and_anno_root + 'annotations.json'
annFile = img_and_anno_root + 'annotations_pingshen.json'
img_save_path = img_and_anno_root + 'crop_img'

if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

def cut_image(boxes, labels, image):
    global count
    for box in boxes:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[0])+int(box[2])
        y2 = int(box[1])+int(box[3])
        img = image[y1:y2, x1:x2, :]
        cv2.imwrite(os.path.join(img_save_path, '{}.jpg'.format(count)),  img)
        count += 1
    pass

'''
    for box, label in zip(boxes, labels):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0]) + int(box[2]), int(box[1]) +int(box[3])), (0, 255, 0), 2)   # xywh，需要转成左上角坐标, 右下角坐标

        # clw note:主要用于输出中文的label
        cv2img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(cv2img)
        draw = ImageDraw.Draw(pilimg)  # 图片上打印
        #font = ImageFont.truetype('/media/clwclw/data/fonts/simsun.ttf', 20, encoding="utf-8")
        font = ImageFont.truetype('./simsun.ttf', 36, encoding="utf-8")
        #font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 36, encoding="utf-8")

        left = float(box[0])
        top = float(box[1])
        right = float(box[0]) + float(box[2])
        down = float(box[1]) + float(box[3])

        draw.text(((left + right) / 2.0, (top + down) / 2.0), label, (255, 0, 0), font=font) # clw note：在中心位置输出标签

        image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

    return image
'''

# 初始化标注数据的 COCO api
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
cats = sorted(cats, key = lambda e:e.get('id'),reverse = False) # clw note：这里并不完全是COCO格式，只能算是类COCO格式，因此
                                                                #           比如这里的categories就不是排序的，因此需要手动排序

img_list = os.listdir(img_path)

# 查看所有图片
flag_CUT = False
imgids = coco.getImgIds()
for i, imgId in enumerate(imgids):
    img = coco.loadImgs(imgId)[0]
    image_name = img['file_name']
    print('clw: already read {} images, image_name: {}'.format(i + 1, image_name))
    # print(img)

    anns = coco.imgToAnns[imgId]
    # print(anns)

    coordinates = []
    labels = []
    img = cv2.imread(os.path.join(img_path, image_name))

    special_class_num = 0  # clw note: only see some special class
    for j, ann in enumerate(anns):
        # if ann['category_id'] != 4:  # clw note: only see some special class
        #     continue
        #special_class_num += 1
        # 1、求坐标
        coordinate = []
        # coordinate.append(anns[j]['bbox'][0])
        # coordinate.append(anns[j]['bbox'][1]+anns[j]['bbox'][3])
        # coordinate.append(anns[j]['bbox'][0]+anns[j]['bbox'][2])
        # coordinate.append(anns[j]['bbox'][1])
        coordinate.append(anns[j]['bbox'][0])
        coordinate.append(anns[j]['bbox'][1])
        coordinate.append(anns[j]['bbox'][2])
        coordinate.append(anns[j]['bbox'][3])
        # print(coordinate)
        coordinates.append(coordinate)

        # 2、找到对应的标签
        labels.append(cats[anns[j]['category_id'] - 1 ]['name']) # clw note: category_id start from 1, not 0; the reason to -1 is that the first is 1 but the index should be 0
        #labels.append(cats[anns[j]['category_id']]['name'])   # clw note: category_id start from  0;

        img = cut_image(coordinates, labels, img)

        #if cats[anns[j]['category_id'] - 1]['id']== 13:
        #if cats[anns[j]['category_id'] ]['id'] == 0:
        #    flag_CUT = True

    #if not flag_CUT:
    #    continue
    #flag_CUT = False


