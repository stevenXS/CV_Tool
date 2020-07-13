import os
from tqdm import tqdm
import numpy as np
import cv2
import math

class_list = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field',
              'small-vehicle', 'large-vehicle', 'ship',
              'tennis-court', 'basketball-court',
              'storage-tank', 'soccer-ball-field',
              'roundabout', 'harbor',
              'swimming-pool', 'helicopter',
              'helipad',  'airport', 'container-crane']

def save_to_yolo_format(txt_data, save_path):
    # 1、格式、坐标变换
    if not 'imagesource:GoogleEarth' in txt_data[0] or not 'gsd:' in txt_data[1]:
        raise Exception('Error: the format is not uniform!')

    img = cv2.imread(save_path[:-4] + '.png')
    img_h, img_w = img.shape[:2]
    with open(save_path, 'w') as f:
        for box_info in txt_data[2:]:   # clw note：DOTA原始数据前两行是其他信息，即上面的gsd等，这里需要跳过
                                        #           box_info: [x1, y1, x2, y2, x3, y3, x4, y4, class_name, difficult]
            box_info_list = box_info.strip().split(' ')
            box = np.int0([float(i) for i in box_info_list[:-2]])  # 最后一个值是label对应的index，这里坐标换算要把index如'3'去掉；
            box = box.reshape([4, 2])
            # print(box)
            rect1 = cv2.minAreaRect(box)  # 生成最小外接矩形，返回一个Box2D结构rect：
                                          # （最小外接矩形的中心（x，y），（宽度，高度），旋转角度）
                                          # 如用上面的输入，会输出((15.0, 15.0), (10.0, 10.0), -90.0)
                                          # 旋转角度θ是水平轴（x轴）逆时针旋转，与碰到的矩形的第一条边的夹角
            # print('rect1 = ', rect1)
            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            ### opencv表示法 -> 角度是指长边和x轴所成夹角，从[-90° ~ 90°)
            if w < h:
                theta = -(90+theta)
                w, h = h, w
            elif w >= h:
                theta = -theta
            theta = theta / 360 * 2 * math.pi
            ###

            bbox_info_str = ""
            for item in [ class_list.index(box_info_list[-2]), x / img_w, y / img_h, w / img_w, h / img_h, theta ]:
                bbox_info_str += (str(item) + ' ')
            bbox_info_str += '\n'

            # 2、写入txt
            f.write(str(bbox_info_str))



if __name__ == '__main__':

    label_dir= 'D:/dataset/DOTA/origin_label/DOTA-v1.5_val'
    save_dir = 'D:/dataset/DOTA/image'   # save txt label in image dir

    image_filenames = [i for i in os.listdir(save_dir) if 'png' in i]  # png is the default format of DOTA, not jpg
    label_filenames = [i for i in os.listdir(label_dir) if 'txt' in i]

    print ('find image: ', len(image_filenames))
    print ('find label: ', len(label_filenames))

    with open('train.txt', 'w') as f:
        for idx, label_filename in enumerate(tqdm(label_filenames)):
            # 1、在数据集图片目录下生成一一对应的 txt 文件
            txt_data = open(os.path.join(label_dir, label_filename), 'r').readlines()
            label_path = os.path.join(save_dir, label_filename)
            label_path = label_path.replace('\\', '/')
            save_to_yolo_format(txt_data, label_path)

            # 2、同时在当前目录生成yolov3所需的文件路径列表
            f.write(label_path[:-4] + '.png' + '\n')




