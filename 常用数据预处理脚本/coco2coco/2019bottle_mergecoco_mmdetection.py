# -*- encoding:utf-8 -*-
# @Time    : 2019/8/31 17:52
# @Author  : gfjiang
# @Site    :
# @File    : merge_coco.py
# @Software: PyCharm
import copy
import json
import os
import os.path as osp

def makedirs(path):
    """对os.makedirs进行扩展

    从路径中创建文件夹，可创建多层。如果仅是文件名，则无须创建，返回False；
    如果是已存在文件或路径，则无须创建，返回False

    Args:
        path: 路径，可包含文件名。纯路径最后一个字符需要是os.sep
    """
    if path is None or path == '':  # 空
        return False
    if osp.isfile(path):    # 是文件并且已存在
        return False
    # 不能使用os.sep，因为有时在windows平台下用户也会传入使用'/'分割的路径
    if '/' not in path and '\\' not in path:  # 不含路径
        return False
    path = osp.dirname(path)
    if osp.exists(path):
        return False
    try:
        os.makedirs(path)
    except Exception as e:
        print(e, 'make dirs failed!')
        return False
    return True

# 加载json文件
def load_json(file):
    """加载json文件

    Args:
        file: 包含路径的文件名

    Returns:

    """
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def dump_json(data, to_file='data.json'):
    """写json文件

    Args:
        data: 待保存成json格式的对象
        to_file: 保存的文件名
    """
    # save json format results to disk
    makedirs(to_file)
    with open(to_file, 'w') as f:
        json.dump(data, f)  # using indent=4 show more friendly
    print('!save {} finished'.format(to_file))



class MergeCOCO(object):
    """merge multiple coco-like datasets into one file
    Args:
        files (list): a list of str or COCO object
    """
    def __init__(self, files):
        if not isinstance(files, (list, tuple)):
            raise TypeError('files must be a list, but got {}'.format(
                type(files)))
        assert len(files) > 1, 'least 2 files must be provided!'
        self.files = files
        if isinstance(self.files[0], dict):
            self.merge_coco = copy.deepcopy(self.files[0])
        else:
            self.merge_coco = load_json(self.files[0])
        self.img_ids = [img_info['id']
                        for img_info in self.merge_coco['images']]
        self.ann_ids = [img_info['id']
                        for img_info in self.merge_coco['annotations']]

    def update_img_ann_ids(self, images, anns):
        img_id_map = dict()
        img_max_id = max(self.img_ids)
        for i in range(len(images)):
            img_id_map[images[i]['id']] = \
                images[i]['id'] + img_max_id + 1
            images[i]['id'] += img_max_id + 1
        self.merge_coco['images'] += images

        ann_max_id = max(self.ann_ids)
        for i in range(len(anns)):
            anns[i]['id'] += ann_max_id + 1
            new_img_id = img_id_map[anns[i]['image_id']]
            anns[i]['image_id'] = new_img_id
            self.ann_ids.append(anns[i]['id'])
        self.merge_coco['annotations'] += anns

    def merge(self, to_file=None):
        for dataset in self.files[1:]:
            if not isinstance(dataset, dict):
                dataset = load_json(dataset)
            self.update_img_ann_ids(
                dataset['images'], dataset['annotations'])
        if to_file:
            self.save(save=to_file)
        return self.merge_coco

    def save(self, save='merge_coco.json'):
        dump_json(self.merge_coco, save)


if __name__ == '__main__':
    # clw note ： 把categories信息更全的放在前面
    dataset_files = ['C:/Users/62349/Downloads/chongqing1_round2_train_20200213/val.json','C:/Users/62349/Downloads/chongqing1_round1_train_20191223/val.json']
    merge_coco = MergeCOCO(dataset_files)
    merge_coco.merge()
    merge_coco.merge_coco['categories']= sorted(merge_coco.merge_coco['categories'], key=lambda e: e['id'], reverse=False)
    merge_coco.save(save='val.json')