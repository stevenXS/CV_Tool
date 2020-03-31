import cv2
import os
import numpy as np
import time

img_path = '/media/clwclw/data/2019bottle/jiuye/images'
save_path = '/media/clwclw/data/2019bottle/jiuye/images_processed6'
if not os.path.exists(save_path):
    os.makedirs(save_path)
img_names = os.listdir(img_path)
img_nums = len(img_names)

for i, img_name in enumerate(img_names):
    print('clw: idx:', i)
    if 'imgs' in img_name and '_0.jpg' in img_name:
        lists = img_name.split("_")
        name = "{}_{}".format(lists[0], lists[1])

        # read 5 images in one group, and calculate mean value of 5 images to be a template
        imgs = []
        mean_img = 0
        for i in range(5):
            img = cv2.imread(os.path.join(img_path, "{}_{}.jpg".format(name, i)), cv2.IMREAD_GRAYSCALE)
            mean_img += img.astype(np.int16)  # clw modify
            # mean_img += img
            imgs.append(img)
        mean_img = mean_img // 5
        mean_img = mean_img.astype(np.uint8)  # clw modify
        imgs_mean = []

        # 4 stage diff, to reduce noise
        tmps_1 =[]
        tmps_2 = []
        tmps_3 = []
        tmps = []

        tmp1_1 = cv2.subtract(imgs[1], imgs[0]) # detect what, substract what，leave a white point
        tmp1_2 = cv2.subtract(imgs[2], imgs[1])
        tmp1_3 = cv2.subtract(imgs[3], imgs[2])
        tmp1_4 = cv2.subtract(imgs[4], imgs[3])
        tmp1_5 = cv2.subtract(imgs[0], imgs[4])
        tmps_1.append(tmp1_1)
        tmps_1.append(tmp1_2)
        tmps_1.append(tmp1_3)
        tmps_1.append(tmp1_4)
        tmps_1.append(tmp1_5)

        tmp2_1 = cv2.subtract(tmp1_1, tmp1_2)
        tmp2_2 = cv2.subtract(tmp1_2, tmp1_3)
        tmp2_3 = cv2.subtract(tmp1_3, tmp1_4)
        tmp2_4 = cv2.subtract(tmp1_4, tmp1_5)
        tmp2_5 = cv2.subtract(tmp1_5, tmp1_1)
        tmps_2.append(tmp2_1)
        tmps_2.append(tmp2_2)
        tmps_2.append(tmp2_3)
        tmps_2.append(tmp2_4)
        tmps_2.append(tmp2_5)

        tmp3_1 = cv2.subtract(tmp2_1, tmp2_2)
        tmp3_2 = cv2.subtract(tmp2_2, tmp2_3)
        tmp3_3 = cv2.subtract(tmp2_3, tmp2_4)
        tmp3_4 = cv2.subtract(tmp2_4, tmp2_5)
        tmp3_5 = cv2.subtract(tmp2_5, tmp2_1)
        tmps_3.append(tmp3_1)
        tmps_3.append(tmp3_2)
        tmps_3.append(tmp3_3)
        tmps_3.append(tmp3_4)
        tmps_3.append(tmp3_5)


        tmp4_1 = cv2.subtract(tmp3_1, tmp3_2)
        tmp4_2 = cv2.subtract(tmp3_2, tmp3_3)
        tmp4_3 = cv2.subtract(tmp3_3, tmp3_4)
        tmp4_4 = cv2.subtract(tmp3_4, tmp3_5)
        tmp4_5 = cv2.subtract(tmp3_5, tmp3_1)
        tmp4_1 = tmp4_1[:, :, np.newaxis]
        tmp4_2 = tmp4_2[:, :, np.newaxis]
        tmp4_3 = tmp4_3[:, :, np.newaxis]
        tmp4_4 = tmp4_4[:, :, np.newaxis]
        tmp4_5 = tmp4_5[:, :, np.newaxis]
        tmps.append(tmp4_1)
        tmps.append(tmp4_2)
        tmps.append(tmp4_3)
        tmps.append(tmp4_4)
        tmps.append(tmp4_5)


        # concate
        mean_img = mean_img[:, :, np.newaxis]
        for i in range(5):
            imgs[i] = imgs[i][:, :, np.newaxis]
            a, b, c = i, i+1, i+2
            if i > 2:
                b, c = b - 4, c - 4
            #img = np.concatenate((imgs[i], tmps[i], mean_img), axis=2)
            img = np.concatenate((tmps[a], tmps[a], tmps[a]), axis=2)

            dst_path = os.path.join(save_path, "{}_{}.jpg".format(name, i))
            cv2.imwrite(dst_path, img)

print("end")