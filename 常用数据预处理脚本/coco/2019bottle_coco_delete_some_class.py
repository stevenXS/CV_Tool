import os
import json

#src_json_file = "C:/Users/62349/Downloads/chongqing1_round2_train_20200213/annotations.json"
#src_json_file = "C:/Users/62349/Downloads/chongqing1_round1_train_20191223/annotations_origin.json"
src_json_file = "C:/Users/62349/Downloads/val.json"
dst_json = 'val.json'

with open(src_json_file, 'r') as f:
    label_data = json.load(f)
categ_infos = label_data['categories']
annot_data = label_data['annotations']
images_info = label_data['images']  # img_name, img_id, img_height, img_width

# delet bg class
new_images_data = []
new_annot_data = []
new_img_id = 0
box_id = 0
select_width = [658, 4096]
select_ids = [1, 2, 3, 4, 5, 9, 10, 12, 13]  # clw modify：ann要保留哪些类别的bbox，没有的就是要删除的
select_ids_for_categorys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # clw modify: json的categories里面要包含的类
new_categ_infos = [categ_info for categ_info in categ_infos if categ_info['id'] in select_ids_for_categorys]   # delet the bg class
for data in images_info:
    if data['width'] in select_width:
        new_img_id += 1
        img_id = data['id']
        for ann_data in annot_data:
            ann_img_id = ann_data['image_id']
            if ann_img_id == img_id:  # match
                if ann_data['category_id'] in select_ids:    # skip when = 0
                    box_id += 1
                    new_ann_data = ann_data.copy()
                    new_ann_data['image_id'] = new_img_id
                    new_ann_data['id'] = box_id
                    new_annot_data.append(new_ann_data)
                else:
                    print(ann_data)
        new_data = data.copy()
        new_data['id'] = new_img_id
        new_images_data.append(new_data)

new_out_json_data = {}
new_out_json_data['info'] = label_data['info']
new_out_json_data['license'] = label_data['license']
new_out_json_data['categories'] = new_categ_infos
new_out_json_data['images'] = new_images_data
new_out_json_data['annotations'] = new_annot_data

with open(dst_json, 'w') as f:
    json.dump(new_out_json_data, f)
