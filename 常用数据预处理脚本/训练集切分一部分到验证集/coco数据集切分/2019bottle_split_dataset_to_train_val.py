# written by clw
import json

# 输入src_json_file，
# 输出train.json, val.json
src_json_file = "C:/Users/62349/Downloads/chongqing1_round2_train_20200213/annotations.json"

with open(src_json_file, 'r') as f:
    label_data = json.load(f)

categ_infos = label_data['categories']
annot_data = label_data['annotations']  # {'area': 2993.0, 'iscrowd': 0, 'image_id': 1, 'bbox': [2500.0, 1268.0, 41.0, 73.0], 'category_id': 12, 'id': 1}
images_info = label_data['images']  # img_name, img_id, img_height, img_width


category_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
img_format = '.jpg'

'''
函数功能：
输入:json中的annotations信息列表，[{'area': 2993.0, 'iscrowd': 0, 'image_id': 1, 'bbox': [2500.0, 1268.0, 41.0, 73.0], 'category_id': 12, 'id': 1}, {....}, ....]
输出:每一类缺陷数量的统计结果，以dict的形式返回， 如{12: 73, 13: 131, 11: 2469}
'''
def count_gt_nums(annot_data):
    gt_nums_count = {}
    for idx, annot in enumerate(annot_data):
        if annot['category_id'] not in category_ids:
            print('error: class %d in image_id %s not exist in zh_dict' % (annot['category_id'], annot['image_id']))

        if annot['category_id'] in gt_nums_count:
            gt_nums_count[annot['category_id']] += 1
        else:
            gt_nums_count[annot['category_id']] = 1
    return gt_nums_count


# 1、统计整个数据集拥有的每个类别的缺陷数量
gt_nums_count_all = count_gt_nums(annot_data)
print('gt nums:', gt_nums_count_all)


# 2、随机按一定比例（默认8:2，即验证集占20%）来划分训练集和验证集
#    比如数据集一共有100张图片，随机抽20张作为验证集
#    对验证集的类别数量进行统计，如果不能保证每一类缺陷的个数都在20%左右的一定范围，如19%-21%
#    则回到该步骤开始的地方，重新划分，重新统计，如此往复...
import random
random.seed(0)
assert round(random.random(), 3) == 0.844, '随机数种子不同，导致不同环境下生成的验证集不同！'

img_nums = len(images_info)  # 统计总共的图片个数
val_split_ratio = 0.2
class_split_ratio_min = 0.19   # 每一类在验证集的object个数不能少于0.15，
class_split_ratio_max = 0.21  # 每一类在验证集的object个数不能多于0.25
count_split_times = 0  # 统计切分验证集的随机次数

while(1):
    bsuccessed_split = True

    # 注意这里不能直接随机选择标注框，annot_data_val = annot_data[:int(val_split_ratio * img_nums)]，而要随机选image_id
    val_index = sorted(random.sample(range(img_nums), int(val_split_ratio * img_nums)))
    train_index = list(set(range(img_nums)).difference(val_index))
    images_info_val = [images_info[i] for i in val_index]
    images_info_train = [images_info[i] for i in train_index]

    annot_data_val = []
    annot_data_train = []
    for i, ann in enumerate(annot_data):
        if ann["image_id"] in val_index:
            annot_data_val.append(ann)
        elif ann["image_id"] in train_index:
            annot_data_train.append(ann)
        else:
            assert False, 'image_id has some bug, not in val_index and train_index'

    gt_nums_count_val = count_gt_nums(annot_data_val)
    for key in gt_nums_count_all:
        assert gt_nums_count_all[key] > 10, '每个类别都不能少于10个样本，否则认为数据及有问题'
        if key not in gt_nums_count_val:  # 如果验证集里面都没有这个key，说明不行，直接重新分验证集
            bsuccessed_split = False
            break

        # 如果某个类别满足条件，则继续下一个类别；否则直接重新分验证集；直到所有类别都满足条件，才可以切分
        if gt_nums_count_val[key] < gt_nums_count_all[key] * class_split_ratio_max and gt_nums_count_val[key] > gt_nums_count_all[key] * class_split_ratio_min:
            continue
        else:
            bsuccessed_split = False
            count_split_times += 1
            print('切分验证集经过的随机碰撞次数：', count_split_times)
            break

    if bsuccessed_split == False:  # 如果经过上面的切分，失败了，那么重新回到while循环开始
        continue
    else:                          # 满足切分条件
        print('切分验证集满足每个类别缺陷个数在%f~%f的范围要求，切分验证集经过的随机碰撞次数：%d' % (class_split_ratio_min, class_split_ratio_max, count_split_times))
        print('数据集总的图片数量:', img_nums)
        print('验证集图片数量：', int(val_split_ratio * img_nums))
        print('数据集总的gt数量:', gt_nums_count_all)
        print('验证集总的gt数量:', gt_nums_count_val)

        # 3 生成train.json和val.json两个文件
        label_data['categories'] = categ_infos
        label_data['annotations'] = images_info_val
        label_data['images'] = annot_data_val
        with open('val.json', 'w') as fp:
            json.dump(label_data, fp, indent=1, separators=(',', ': '))

        label_data['categories'] = categ_infos
        label_data['annotations'] = images_info_train
        label_data['images'] = annot_data_train
        with open('train.json', 'w') as fp:
            json.dump(label_data, fp, indent=1, separators=(',', ': '))

        break

print('end!')