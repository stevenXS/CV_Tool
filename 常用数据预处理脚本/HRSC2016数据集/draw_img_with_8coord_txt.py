import cv2
import os
import torch
import numpy as np
import random

def plot_one_box(x, img, color=None, label=None, line_thickness=None):  # clw note: x is tensor
    if isinstance(x, np.ndarray) or isinstance(x, list):
        x = torch.tensor(x, dtype=torch.float32)
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    coor = torch.cuda.FloatTensor(x).reshape(4,2) if x.is_cuda else torch.FloatTensor(x).reshape(4,2)
    cv2.polylines(img, [coor.cpu().numpy().astype(np.int32)], True, color, tl)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # label_coor = get_rotated_coors(torch.FloatTensor([coor[0,0], coor[0,1], t_size[0], t_size[1], x[-1]]))
        # #cv2.polylines(img,[label_coor.reshape(4,2).cpu().numpy().astype(np.int32)],True, color, tl)
        cv2.putText(img, label, tuple(coor[0].cpu().numpy()) , 0, tl / 3, color, thickness=tf, lineType=cv2.LINE_AA)


#img_path = 'D:/dataset/hrsc2016/train'
img_path = 'D:/dataset/HRSC2016_dataset/HRSC2016/Train/hrsc2016'

img_format = '.jpg'
img_names = [name for name in os.listdir(img_path) if name.endswith(img_format)]

for img_name in img_names:
    img = cv2.imread(os.path.join(img_path, img_name))
    img_h, img_w, _ = img.shape
    with open(os.path.join(img_path, img_name[:-4] + '.txt'), 'r') as f:
        labels = f.readlines()
        for label in labels:
            box = label.strip().split(' ')[1:]
            x0 = int(img_w * float(box[0]))
            y0 = int(img_h * float(box[1]))
            x1 = int(img_w * float(box[2]))
            y1 = int(img_h * float(box[3]))
            x2 = int(img_w * float(box[4]))
            y2 = int(img_h * float(box[5]))
            x3 = int(img_w * float(box[6]))
            y3 = int(img_h * float(box[7]))
            plot_one_box([x0, y0, x1, y1, x2, y2, x3, y3], img)

    cv2.imshow('aaa', img)
    cv2.waitKey(0)

