import os
import cv2
import albumentations as A

root_path = '/media/clwclw/data/2019bottle/jiuye/'
img_path = os.path.join(root_path, 'images_processed4')
img_names = os.listdir(img_path)
save_path = os.path.join(root_path, 'images_processed6')
if not os.path.exists(save_path):
    os.mkdir(save_path)

for i, img_name in enumerate(img_names):
    print(i+1)
    img = cv2.imread(os.path.join(img_path, img_name))
    augmentation =  A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0, p=1)
    data = {"image": img}
    augmented = augmentation(**data)
    cv2.imwrite(os.path.join(save_path, img_name), augmented['image'])