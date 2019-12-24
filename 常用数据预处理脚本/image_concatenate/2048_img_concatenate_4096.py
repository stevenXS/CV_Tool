#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ! python3
import cv2
import numpy as np
import os
import glob

merge_image_count = 0
def merge_image(img_paths):
    global merge_image_count
    for i in range(len(img_paths) - 1):
        print('clw: merged %d nums of images' % (merge_image_count + 1))
        merge_image_count += 1
        img1 = cv2.imread(img_paths[i])
        img2 = cv2.imread(img_paths[i + 1])
        # ====使用numpy的数组矩阵合并concatenate======

        img = np.vstack((img1, img2))
        # 纵向连接 image = np.vstack((gray1, gray2))
        # 横向连接 image = np.concatenate([gray1, gray2], axis=1)
        img_paths[i] = img_paths[i].replace('\\', '/')
        cv2.imwrite(
            os.path.join(img_save_path, img_paths[i].split('/')[-1].split('.')[0] + '_merged_4096' + img_format), img)


img_format = '.jpg'
img_folder_path = 'C:/Users/62349/Desktop/captured/'
img_paths_GC1 = glob.glob(os.path.join(img_folder_path, '*GC1*' + img_format))
img_paths_GC2 = glob.glob(os.path.join(img_folder_path, '*GC2*' + img_format))
img_paths_GC4 = glob.glob(os.path.join(img_folder_path, '*GC4*' + img_format))

img_save_path = 'C:/Users/62349/Desktop/captured/merged'
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

merge_image(img_paths_GC1)
merge_image(img_paths_GC2)
merge_image(img_paths_GC4)



print('end!')