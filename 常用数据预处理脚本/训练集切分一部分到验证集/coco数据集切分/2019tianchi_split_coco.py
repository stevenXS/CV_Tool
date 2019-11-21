# clw note: split the dataset random by the val propotion factor

import json
from collections import Counter

anno_path = '/mfs/home/fangyong/data/guangdong/round2/train/Annotations/train.json'
save_train_path = '/mfs/home/fangyong/data/guangdong/round2/train/Annotations/train_.json'
save_val_path = '/mfs/home/fangyong/data/guangdong/round2/train/Annotations/val_.json'

annos = json.load(open(anno_path, 'rb'))
ann_list = annos['annotations']
defect_names = []
for data in ann_list:
    defect_names.append(data['category_id'])

img_list = annos['images']
image_ids = []
for data in img_list:
    image_ids.append(data['id'])

train_annos = []
val_annos = []

defect_nums = Counter(defect_names)
defect_counts = {}

val_propotion_factor = 0.05

import random
val_random_list = random.sample(image_ids, int(len(img_list) * val_propotion_factor))
for i, anno in enumerate(annos['annotations']):  # clw note: image_id in annos['annotations'] usually(and must) be sorted
    if int(anno['image_id']) not in val_random_list:
        train_annos.append(anno)
    else:
        val_annos.append(anno)

### Step 2: filter the image_id in val_random_list
val_images = []
train_images = []
for i, image_info in enumerate(annos['images']):
    if image_info['id'] in val_random_list:
        val_images.append(image_info)
    else:
        train_images.append(image_info)

### Step 3: save to json
annos['annotations'] = train_annos
annos['images'] = train_images
with open(save_train_path, 'w') as fp:
    json.dump(annos, fp, indent=1, separators=(',', ': '))

annos['annotations'] = val_annos
annos['images'] = val_images
with open(save_val_path, 'w') as fp:
    json.dump(annos, fp, indent=1, separators=(',', ': '))

print('clw: -------------end!-------------')



