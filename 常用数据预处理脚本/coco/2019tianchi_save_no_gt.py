# written by teammate chenzhichao

import os
import json
import glob
import cv2
if __name__ == '__main__':
    #base_path = '/data2/czc/data/tc/round2/data/defect/normal'
    base_path = '/mfs/home/fangyong/data/guangdong/round2/train2/normal'
    subdir_list = glob.glob(os.path.join(base_path, '**'))
    base_id = 1263 # # clw note: To add to the json file, you need to see the final_id(+1) of the json and then copy this json's content to it.
    out_labels = []
    data_length = len(subdir_list)
    for index, subdir_full_name in enumerate(subdir_list):
        subdir_name = subdir_full_name.split('/')[-1]
        img_name = subdir_name + '.jpg'
        img_full_path = os.path.join(subdir_full_name, img_name)
        height, width, _ = cv2.imread(img_full_path).shape
        img_name_label = os.path.join('normal', subdir_name, img_name)
        base_id += 1
#         print('{},{},{},{}'.format(height, width, base_id, img_name_label))
        out_label = dict()
        out_label['height'] = height
        out_label['width'] = width
        out_label['id'] = base_id
        out_label['file_name'] = img_name_label
        out_labels.append(out_label)
        print('process {}/{}'.format(index, data_length))
    save_path = 'norm.json'
    with open(save_path, 'w') as fp:
        json.dump(out_labels, fp, indent=1, separators=(',', ': '))
