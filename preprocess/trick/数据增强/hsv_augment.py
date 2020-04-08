### written by clw, reference ultralytics/yolov3

import cv2
import numpy as np

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    x = (np.random.uniform(-1, 1, 3) * np.array([hgain, sgain, vgain]) + 1).astype(np.float32)  # random gains
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x.reshape((1, 1, 3))).clip(None, 255).astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_hsv

def augment_h(img, hgain=1):
    hsv_scale = np.array([hgain, 1, 1])  # random gains
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * hsv_scale.reshape((1, 1, 3))).clip(None, 255).astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img_hsv)  # no return needed
    return img_hsv

def augment_s(img, sgain=1):
    hsv_scale = np.array([1, sgain, 1])  # random gains
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * hsv_scale.reshape((1, 1, 3))).clip(None, 255).astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img_hsv)  # no return needed
    return img_hsv

def augment_v(img, vgain=1):
    hsv_scale = np.array([1, 1, vgain])  # random gains
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * hsv_scale.reshape((1, 1, 3))).clip(None, 255).astype(np.uint8)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img_hsv)
    return img_hsv

if __name__ == "__main__":
    img_path = 'C:/Users/Administrator/Desktop/2.jpg'
    img = cv2.imread(img_path)
    # for i in range(10):
    #     augment_hsv(img, hgain=0.0103, sgain=0.691, vgain=0.433)
    #     cv2.imwrite(img_path.split('.')[0] + '_' + str(i) + '.jpg', img)

    gain = 1.5
    mode = 'v'  # or 's' or 'v'

    if mode == 'h':
        img_hsv = augment_h(img, gain) # 修改色调
        cv2.imwrite(img_path.split('.')[0] + '_' + 'h' + str(gain) + '.jpg', img_hsv)
    elif mode == 's':
        img_hsv = augment_s(img, gain) # 修改饱和度
        cv2.imwrite(img_path.split('.')[0] + '_' + 's' + str(gain) + '.jpg', img_hsv)
    elif mode == 'v':
        img_hsv = augment_v(img, gain) # 修改亮度
        cv2.imwrite(img_path.split('.')[0] + '_' + 'v' + str(gain) + '.jpg', img_hsv)
