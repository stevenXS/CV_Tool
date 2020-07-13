#!/usr/bin/env python
# coding=UTF-8

from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
#import pylab
import cv2
import os
from skimage.io import imsave
import numpy as np
from PIL import Image, ImageDraw, ImageFont
#pylab.rcParams['figure.figsize'] = (8.0, 10.0)


img_and_anno_root = 'D:/2020water/train/'
img_path = img_and_anno_root + 'image/'
annFile = img_and_anno_root + 'annotations.json'

img_save_path = img_and_anno_root + 'visualized_image'
VISUALIZE_SINGLE = False
NEED_SAVE = False

if NEED_SAVE:
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

def draw_rectangle(boxes, labels, image):

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


# 初始化标注数据的 COCO api
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())

cats = sorted(cats, key = lambda e:e.get('id'),reverse = False) # clw note：这里并不完全是COCO格式，只能算是类COCO格式，因此
                                                                #           比如这里的categories就不是排序的，因此需要手动排序

img_list = os.listdir(img_path)

# 查看某张图片
if VISUALIZE_SINGLE:
    for i in range(len(img_list)):
    # for i in range(5): # clw note：随机查看几张
        imgIds = i+1
        img = coco.loadImgs(imgIds)[0]
        image_name = img['file_name']
        if image_name != '7e7e6359d8fb8e9c1332319214.jpg':
            continue
        else:
            # catIds=[] 说明展示所有类别的box，也可以指定类别
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
            anns = coco.loadAnns(annIds)
            # print(anns)
            coordinates = []
            labels = []
            img_raw = cv2.imread(os.path.join(img_path, image_name))
            for j in range(len(anns)):
                # 1、求坐标
                coordinate = []
                coordinate.append(anns[j]['bbox'][0])
                coordinate.append(anns[j]['bbox'][1])
                coordinate.append(anns[j]['bbox'][2])
                coordinate.append(anns[j]['bbox'][3])
                coordinates.append(coordinate)

                # 2、找到对应的标签

                labels.append(cats[anns[j]['category_id']]['name'])

            image = draw_rectangle(coordinates, labels, img_raw)
            cv2.namedWindow(image_name, 0);
            cv2.resizeWindow(image_name, 1280, 1024);
            cv2.imshow(image_name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


else:
    # 查看所有图片
    flag_SHOW = True
    imgids = coco.getImgIds()
    for i, imgId in enumerate(imgids):
    #for imgId in range(4106, 4106+len(img_list)):
    #for i in range(3000, len(img_list)):  # look up from xxxx
    # for i in range(5): # clw note：随机查看几张
        img = coco.loadImgs(imgId)[0]
        image_name = img['file_name']
        print('clw: already read {} images, image_name: {}'.format(i+1, image_name))
        # print(img)

        # #加载并显示图片
        # I = io.imread('%s/%s' % (img_path, img['file_name']))
        # plt.axis('off')
        # plt.imshow(I)
        # plt.show()

        # catIds=[] 说明展示所有类别的box，也可以指定类别

        # clw note:读取image_id对应标注信息的两种方法：
        #           method 1, don't fit to this dataset
        # annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
        # anns = coco.loadAnns(annIds)

        # method 2, fit this dataset!!!
        anns = coco.imgToAnns[imgId]
        #print(anns)

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
            labels.append(cats[anns[j]['category_id'] - 1 ]['name'])  # clw note: start from 1, not 0; the reason to -1 is that the first is 1 but the index should be 0
            #labels.append(cats[anns[j]['category_id']]['name'])  # clw note: start from 0;

            img = draw_rectangle(coordinates, labels, img)

            #if cats[anns[j]['category_id'] - 1]['id']== 13:
            #if cats[anns[j]['category_id'] ]['id'] == 0:
            #    flag_SHOW = True

        #if not flag_SHOW:
        #    continue
        #flag_SHOW = False

        if NEED_SAVE:
            cv2.imwrite(os.path.join(img_save_path, image_name), img)
        else:
            cv2.namedWindow(image_name, 0);
            cv2.resizeWindow(image_name, 1024, 576);
            cv2.moveWindow(image_name, 0, 0);
            cv2.imshow(image_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

