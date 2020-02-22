import os
import json

path = 'C:/Users/62349/Downloads/chongqing1_round1_train_20191223'
src_json =  'annotations_pinggai.json'
dst_json = './annotations_pinggai_processed.json'

with open(os.path.join(path, src_json), 'r') as f:
    label_data = json.load(f)
categ_infos = label_data['categories']
annot_data = label_data['annotations']
images_info = label_data['images']  # img_name, img_id, img_height, img_width

categ_infos = sorted(categ_infos, key = lambda e:e.get('id'),reverse = False)
label_data['categories'] = categ_infos

with open(os.path.join(path, dst_json), 'w') as fp:
    json.dump(label_data, fp, indent=1, separators=(',', ': '))

