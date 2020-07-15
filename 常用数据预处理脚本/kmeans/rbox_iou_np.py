import torch
import numpy as np
import cv2
import math
from shapely.geometry import Polygon
import time


# anchor对齐阶段计算iou
def skewiou(box1, box2):
    a=box1.reshape(4, 2)
    b=box2.reshape(4, 2)
    # 所有点的最小凸的表示形式，四边形对象，会自动计算四个点，最后顺序为：左上 左下  右下 右上 左上
    poly1 = Polygon(a).convex_hull
    poly2 = Polygon(b).convex_hull
    if not poly1.is_valid or not poly2.is_valid:
        print('formatting errors for boxes!!!! ')
        return 0
    if  poly1.area == 0 or  poly2.area  == 0 :
        return 0

    inter = Polygon(poly1).intersection(Polygon(poly2)).area
    union = poly1.area + poly2.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

def get_rotated_coors(box):
    assert len(box) > 0 , 'Input valid box!'
    cx = box[0]; cy = box[1]; w = box[2]; h = box[3]; a = box[4]
    xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
    t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(cx,cy), scale=1)
    x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2]
    y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2]
    x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2]
    y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2]
    x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2]
    y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2]
    x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2]
    y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2]

    if isinstance(x0, np.ndarray):
        r_box = np.concatenate([np.expand_dims(x0, 0), np.expand_dims(y0, 0),
                         np.expand_dims(x1, 0), np.expand_dims(y1, 0),
                         np.expand_dims(x2, 0), np.expand_dims(y2, 0),
                         np.expand_dims(x3, 0), np.expand_dims(y3, 0)], 0)
    else:
        r_box = np.array([x0,y0,x1,y1,x2,y2,x3,y3])
    return r_box


# 输入的是xywha
# 支持输入单个的box和多box tensor的iou计算，其中默认box1为单个的
def skew_bbox_iou(box1, box2, wh_iou=False):
    if isinstance(box1, list):  box1 = np.array(box1, dtype=float)
    if isinstance(box2, list):  box2 = np.array(box2, dtype=float)

    if len(box1.shape) < len(box2.shape):  # 输入的单box维度不匹配时，unsqueeze一下
        box1 = np.expand_dims(box1, 0)
    # if not box1.shape == box2.shape:
    #     box1 = np.tile(box1, (len(box2), 1))

    if wh_iou:
        box1_xyctr = np.zeros((box1.shape[0], 2), dtype=float)  # clw note: 如果只给了wh求iou，那么可以假设一个x_ctr和y_ctr，
                                                           #           因为两个是同心矩形，因此设成多少结果都一样
        box2_xyctr = np.zeros((box2.shape[0], 2), dtype=float)
        box1 = np.column_stack((box1_xyctr, box1))
        box2 = np.column_stack((box2_xyctr, box2))
    else:
        box1 = box1[:, :5]
        box2 = box2[:, :5]

    ious_all = []
    for i in range(len(box1)):
        ious = []
        for j in range(len(box2)):
            r_b1 = get_rotated_coors(box1[i])
            r_b2 = get_rotated_coors(box2[j])
            ious.append(skewiou(r_b1, r_b2))
        ious_all.append(ious)
    return np.array(ious_all, dtype=float)


if __name__ == '__main__':
    start = time.time()
    box1 = [[10, 10, 20, 20, 0]]
    box2 = [[20, 10, 14.14, 14.14, math.pi / 4], [10, 10, 20, 20, 0]]
    box3 = [20, 10, 0]
    box4 = [[14.14, 14.14, math.pi / 4]]
    print(skew_bbox_iou(box1, box2))  # iou = 0.2
    print(skew_bbox_iou(box3, box4, wh_iou=True))  # iou = 0.6
    print('time use：', time.time() - start)

