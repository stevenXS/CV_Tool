import math
import cv2
import numpy as np
import torch

### clw note：旋转矩形，5坐标转8坐标
# def get_rotated_coors(box):
#     assert len(box) > 0 , 'Input valid box!'
#     cx = box[0]; cy = box[1]; w = box[2]; h = box[3]; a = box[4]
#     xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
#     t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
#     R = np.eye(3)
#     R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(cx,cy), scale=1)
#     x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2]
#     y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2]
#     x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2]
#     y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2]
#     x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2]
#     y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2]
#     x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2]
#     y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2]
#
#     if isinstance(x0,torch.Tensor):
#         r_box=torch.cat([x0.unsqueeze(0),y0.unsqueeze(0),
#                          x1.unsqueeze(0),y1.unsqueeze(0),
#                          x2.unsqueeze(0),y2.unsqueeze(0),
#                          x3.unsqueeze(0),y3.unsqueeze(0)], 0)
#     else:
#         r_box = np.array([x0,y0,x1,y1,x2,y2,x3,y3])
#     return r_box


### clw note：旋转矩形，5坐标转8坐标
def get_rotated_coors(box):
    assert len(box) > 0 , 'Input valid box!'
    cx = box[0]; cy = box[1]; w = box[2]; h = box[3]; a = box[4]
    ####
    if a > 0:  # 和opencv定义方式不一样
        a = -(math.pi/2 - a)
        w, h = h, w
    ##############

    rect = ((cx, cy), (w, h), a*180/math.pi)
    box_8coord = cv2.boxPoints(rect)  # (4, 2)
    r_box = box_8coord.reshape(1, 8).squeeze()
    return r_box