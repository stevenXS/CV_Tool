#coding: utf-8
### 程序功能：数据格式为xml，转成切图后的xml，做一下坐标变换

import os
import numpy as np
import copy, cv2

cropped_width = 1024
cropped_height = 1024
# cropped_width = 512
# cropped_height = 512
step = 1024
img_format = '.jpg'

# 配置好训练集的路径
img_root = "C:/Users/62349/Desktop/111/333/"
#img_root = "C:/Users/62349/Downloads/chongqing1_round1_train1_20191223_split/visualized_image2/"
save_root = img_root
crop_root = save_root + "crop/"
scale_root = save_root + "scale/"
diff_root = save_root + "diff/"

def prepare_folder():
    if not os.path.exists(save_root): 
        os.makedirs(save_root)
    if not os.path.exists(crop_root):
        os.makedirs(crop_root)
    if not os.path.exists(scale_root):
        os.makedirs(scale_root)
    if not os.path.exists(diff_root):
        os.makedirs(diff_root)
    pass

# def image_crop(img, img_file_name):  # written by huminglong, but not good
#     row = 0
#     img_name_format = '%s_%04d_%04d' + img_format
#     while row < img.shape[0]:
#         col = 0
#         while col < img.shape[1]:
#             w = cropped_width if img.shape[1] >= col + cropped_width else img.shape[1] - col
#             h = cropped_height if img.shape[0] >= row + cropped_height else img.shape[0] - row
#             img_slice = img[row:row+h, col:col+w]
#             crop_img_path = os.path.join(crop_root, img_name_format % (img_file_name.replace(img_format, ""), row, col))
#             cv2.imwrite(crop_img_path, img_slice)
#             print("save", crop_img_path)
#             col = col+w
#         row = row+h
#     pass

### 裁剪图片 ###
def image_crop(image, img_file_name):
    shape = image.shape
    for start_h in range(0, shape[0], step):
        for start_w in range(0, shape[1], step):
            start_h_new = start_h
            start_w_new = start_w
            if start_h + cropped_height > shape[0]:
                start_h_new = shape[0] - cropped_height  # 如果加了cropped_height超出边界，则回退
            if start_w + cropped_width > shape[1]:
                start_w_new = shape[1] - cropped_width
            top_left_row = max(start_h_new, 0)  # 防止原图本来就没有cropped_heigh大，导致回退后变成负数，所以这里确保不是负数
            top_left_col = max(start_w_new, 0)
            bottom_right_row = min(start_h + cropped_height, shape[0]) # 防止截取的左下角和右下角超边界，因此同样需要约束
            bottom_right_col = min(start_w + cropped_width, shape[1])
            subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]
            img_save_path = os.path.join(crop_root, "%s_%04d_%04d%s" % (img_file_name[:-4], top_left_row, top_left_col, img_format))
            cv2.imwrite(img_save_path, subImage)


def image_scale(img, img_file_name):
    scaled_img = cv2.resize(img,(cropped_width, cropped_height), interpolation=cv2.INTER_AREA)
    scaled_img_path = os.path.join(scale_root, img_file_name)
    cv2.imwrite(scaled_img_path, scaled_img)
    print("save", scaled_img_path)
    pass

def img_rgb_mean(img):
    B_mean = np.mean(img[:,:,0])
    G_mean = np.mean(img[:,:,1])
    R_mean = np.mean(img[:,:,2])
    return R_mean, G_mean, B_mean

def rgb_mean_demo():
    a = np.random.randn(3,9) * 255
    b = a.reshape(3,3,3)
    img = b.astype(np.uint8)
    print(img)
    print(img_rgb_mean(img))
    # scale_diff(img, None)
    pass

def imgs_rgb_mean(image_path):
    file_names = os.listdir(image_path)
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for file_name in file_names:
        img = cv2.imread(os.path.join(image_path, file_name), 1)
        per_image_Bmean.append(np.mean(img[:,:,0]))
        per_image_Gmean.append(np.mean(img[:,:,1]))
        per_image_Rmean.append(np.mean(img[:,:,2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return R_mean, G_mean, B_mean

def scale_diff(img, img_file_name):
    R_mean, G_mean, B_mean = img_rgb_mean(img)
    mean = np.array([B_mean, G_mean, R_mean]).reshape(1,1,3) 
    diff_img = (img - mean) * 5 + 128
    diff_img_path = os.path.join(diff_root, img_file_name)
    cv2.imwrite(diff_img_path, diff_img)
    print("save", diff_img_path)
    pass

def main():
    ### 主函数 ###

    print("list *%s from %s" % (img_format, img_root))
    image_file_names = [i for i in os.listdir(img_root) if img_format in i] 

    for idx, img_file_name in enumerate(image_file_names):
        print (idx + 1, 'read img', img_file_name)
        img_path = os.path.join(img_root, img_file_name)
        img = cv2.imread(img_path)
        image_crop(img, img_file_name)  # 下面这三句根据需要选择
        #image_scale(img, img_file_name)
        # scale_diff(img, img_file_name)
    pass

if __name__ == "__main__":
    prepare_folder()
    main()
    pass





