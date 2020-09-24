import os
import numpy as np
import shutil
from tqdm import tqdm

### 按照 7:2:1 划分
masks_dir = 'C:/Users/62349/Downloads/ebond/masks'
images_dir = 'D:/ISIC-2017_Training_Data'

img_names = os.listdir(images_dir)  # .jpg
mask_names = os.listdir(masks_dir)  # .png

print(len(mask_names))

train_index = np.random.choice(len(mask_names), size=int(len(mask_names)*0.7), replace=False)
val_test_index = np.setdiff1d(list(range(len(mask_names))), train_index)
train_mask_names = [mask_names[i] for i in train_index]
val_test_mask_names = [mask_names[i] for i in val_test_index]
print(len(train_mask_names))
print(len(val_test_mask_names))

val_index = np.random.choice(val_test_index, size=int(len(val_test_mask_names)* 1/3), replace=False)
test_index = np.setdiff1d(val_test_index, val_index)
val_mask_names = [mask_names[i] for i in val_index]
test_mask_names = [mask_names[i] for i in test_index]
print(len(val_mask_names))
print(len(test_mask_names))

aaa = train_mask_names + val_test_mask_names
assert len(aaa) == len(set(aaa))

# 划分
train_img_path = '/root/ebond/train_images'
val_img_path = '/root/ebond/val_images'
test_img_path = '/root/ebond/test_images'

train_masks_path = '/root/ebond/train_masks'
val_masks_path = '/root/ebond/val_masks'
test_masks_path = '/root/ebond/test_masks'

if not os.path.exists(train_img_path):
    os.makedirs(train_img_path)
if not os.path.exists(val_img_path):
    os.makedirs(val_img_path)
if not os.path.exists(test_img_path):
    os.makedirs(test_img_path)

if not os.path.exists(train_masks_path):
    os.makedirs(train_masks_path)
if not os.path.exists(val_masks_path):
    os.makedirs(val_masks_path)
if not os.path.exists(test_masks_path):
    os.makedirs(test_masks_path)

for mask_name in tqdm(train_mask_names):
    img_name = mask_name[:-17] + '.jpg'
    shutil.move(os.path.join(masks_dir, mask_name), os.path.join(train_masks_path, mask_name))
    shutil.move(os.path.join(images_dir, img_name), os.path.join(train_img_path, img_name))
for mask_name in tqdm(val_mask_names):
    img_name = mask_name[:-17] + '.jpg'
    shutil.move(os.path.join(masks_dir, mask_name), os.path.join(val_masks_path, mask_name))
    shutil.move(os.path.join(images_dir, img_name), os.path.join(val_img_path, img_name))
for mask_name in tqdm(test_mask_names):
    img_name = mask_name[:-17] + '.jpg'
    shutil.move(os.path.join(masks_dir, mask_name), os.path.join(test_masks_path, mask_name))
    shutil.move(os.path.join(images_dir, img_name), os.path.join(test_img_path, img_name))