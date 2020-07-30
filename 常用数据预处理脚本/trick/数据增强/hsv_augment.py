### written by clw

import cv2
import numpy as np
import os

def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    x = (np.array([hgain, sgain, vgain])).astype(np.float32)  # random gains
    img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x.reshape((1, 1, 3))).clip(None, 255).astype(np.uint8)
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_hsv

if __name__ == "__main__":
    save_path = 'D:/wheat_hsv_effect'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_path = 'D:/00b5fefed.jpg'
    #img_path = 'D:/00e903abe.jpg'
    img = cv2.imread(img_path)

    for gain in range(1, 20, 1):
        gain /= 10
        #img_hsv = augment_hsv(img, hgain=gain, sgain=1.0, vgain=1.0)
        #img_hsv = augment_hsv(img, hgain=1.0, sgain=gain, vgain=1.0)
        img_hsv = augment_hsv(img, hgain=1.0, sgain=1.0, vgain=gain)
        cv2.imwrite(os.path.join(save_path, img_path.split('/')[-1][:-4] + '_' + str(gain) + '.jpg'), img_hsv)

