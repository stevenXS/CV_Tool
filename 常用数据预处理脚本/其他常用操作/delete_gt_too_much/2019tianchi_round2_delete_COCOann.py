# clw note: for example, if gt > 500, it may cause OOM, so at first we try to delete the annotations of images
#           which contains too many gt.


import json

anno_path = '/mfs/home/fangyong/data/guangdong/round2/train2/Annotations/train.json'
save_path = '/mfs/home/fangyong/data/guangdong/round2/train2/Annotations/train.json'

annos = json.load(open(anno_path, 'rb'))

delete_limit = 50
count_gt = 0
image_id = 0
delete_img_ids = []  # clw note
skip_flag = False

### Step 1: put the count_gt > some number 's image_name to delete_img_ids list []
for i, anno in enumerate(annos['annotations']):
    if skip_flag == True and image_id == int(anno['image_id']):
        continue
    elif skip_flag == True and image_id != int(anno['image_id']):
        skip_flag = False

    if count_gt >= delete_limit:  # clw note: need modify
        print('clw: image_id = ', image_id)
        img_name = annos['images'][anno['image_id']]['file_name']
        #print('clw: delete image_name = ', img_name)
        delete_img_ids.append(anno['image_id'])
        count_gt = 0
        skip_flag = True
    if image_id != int(anno['image_id']):
        image_id = int(anno['image_id'])
        count_gt = 1
    else:
        count_gt = count_gt + 1


### Step 2: filter the image_id that has more than the delete limit
my_annos = []
for i, anno in enumerate(annos['annotations']):
    if anno['image_id'] in delete_img_ids:
        continue
    else:
        my_annos.append(anno)

my_images = []
for i, image_info in enumerate(annos['images']):
    if image_info['id'] in delete_img_ids:
        continue
    else:
        my_images.append(image_info)


### Step 3: save to json
annos['annotations'] = my_annos
annos['images'] = my_images
with open(save_path, 'w') as fp:
    json.dump(annos, fp, indent=1, separators=(',', ': '))


print('clw: -------------end!-------------')



