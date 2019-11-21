### 作者：白德桃，2019天池纺织品缺陷检测比赛论坛分享
# https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.3.43b46448AmIjUk&postId=71169

'''
@javis
'''
import os
import json
import numpy as np
import shutil
import pandas as pd
import cv2

# defect_name2label = {
#     '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
#     '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
#     '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
#     '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
# }
defect_name2label = {
    '破洞': 1, '水渍': 2, '油渍': 3, '污渍': 4, '三丝': 5, '结头': 6, '花板跳': 7, '百脚': 8, '毛粒': 9,
    '粗经': 10, '松经': 11, '断经': 12, '吊经': 13, '粗维': 14, '纬缩': 15, '浆斑': 16, '整经结': 17, '星跳': 18, '跳花': 19,
    '断氨纶': 20, '稀密档': 21, '浪纹档': 22, '色差档': 23, '磨痕': 24, '轧痕': 25, '修痕': 26, '烧毛痕': 27, '死皱': 28, '云织': 29,
    '双纬': 30, '双经': 31, '跳纱': 32, '筘路': 33, '纬纱不良': 34,
}

defect_name2pinyin = {
    '破洞': 'podong', '水渍': 'shuizi', '油渍': 'youzi', '污渍': 'wuzi', '三丝': 'sansi', '结头':  'jietou', '花板跳': 'huabantiao',
    '百脚': 'baijiao', '毛粒': 'maoli','粗经': 'cujing', '松经': 'songjing', '断经': 'duanjing', '吊经': 'diaojing', '粗维': 'cuwei',
    '纬缩': 'weisuo', '浆斑': 'jiangban', '整经结': 'zhengjingjie', '星跳': 'xingtiao', '跳花': 'tiaohua', '断氨纶': 'duananlun',
    '稀密档': 'ximidang', '浪纹档': 'langwendang', '色差档': 'sechadang', '磨痕': 'mohen', '轧痕': 'yahen', '修痕': 'xiuhen',
    '烧毛痕': 'shaomaohen', '死皱': 'sizhou', '云织': 'yunzhi', '双纬': 'shuangwei', '双经': 'shuangjing', '跳纱': 'tiaosha',
    '筘路': 'koulu', '纬纱不良': 'weishabuliang'
}  # clw note: 为了之后转xml然后在labelImg下看标签，因为labelImg不支持汉字

# defect_name2label = {
#     '沾污': 1, '错花': 2, '水印': 3, '花毛': 4, '缝头': 5, '缝头印': 6, '虫粘': 7,
#     '破洞': 8, '褶子': 9, '织疵': 10, '漏印': 11, '蜡斑': 12, '色差': 13, '网折': 14, '其他': 15
# }


class Fabric2COCO:

    def __init__(self,mode="train"):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.mode =mode
        if not os.path.exists("coco/images/{}".format(self.mode)):
            os.makedirs("coco/images/{}".format(self.mode))

    def to_coco(self, anno_file,img_dir):
        self._init_categories()
        anno_result= pd.read_json(open(anno_file,"r"))
        name_list=anno_result["name"].unique()
        for count, img_name in enumerate(name_list):
            img_anno = anno_result[anno_result["name"] == img_name]
            bboxs = img_anno["bbox"].tolist()
            defect_names = img_anno["defect_name"].tolist()
            assert img_anno["name"].unique()[0] == img_name
            ###
            img_path=os.path.join(img_dir,img_name)
            #img =cv2.imread(img_path)
            #h,w,c=img.shape   # clw note: if you are sure that the img_size is fixed, you can use the code below to save lots of times.
            ###
            h,w=1000,2446
            self.images.append(self._image(img_path,h, w))
            print('clw: already read %d images' % (count))

            #self._cp_img(img_path)

            for bbox, defect_name in zip(bboxs, defect_names):
                label= defect_name2label[defect_name]
                annotation = self._annotation(label, bbox)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'fabric defect'
        instance['license'] = ['none']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def _init_categories(self):
        # for v in range(1, len(defect_name2label) + 1):
        #     print(v)
        #     category = {}
        #     category['id'] = v
        #     category['name'] = str(v)
        #     category['supercategory'] = 'defect_name'
        #     self.categories.append(category)
        for k, v in defect_name2label.items():
            category = {}
            category['id'] = v
            category['name'] = defect_name2pinyin[k]
            category['supercategory'] = 'defect_name'
            self.categories.append(category)

    def _image(self, path,h,w):
        image = {}
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path)
        return image

    def _annotation(self,label,bbox):
        area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        points=[[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = label
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    def _cp_img(self, img_path):
        shutil.copy(img_path, os.path.join("coco/images/{}".format(self.mode), os.path.basename(img_path)))

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        '''coco,[x,y,w,h]'''
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def save_coco_json(self, instance, save_path):
        import json
        with open(save_path, 'w') as fp:
            json.dump(instance, fp, indent=1, separators=(',', ': '))


'''转换有瑕疵的样本为coco格式'''
# img_dir = "/media/clwclw/data/2019tianchi/train_val/defect_Images"
# anno_dir = "/media/clwclw/data/2019tianchi/train_val/Annotations/anno_train.json"
img_dir = "F:/deep_learning/dataset/2019tianchi/round1/train/defect_Images"
anno_dir = "F:/deep_learning/dataset/2019tianchi/round1/train/Annotations/anno_train.json"
fabric2coco = Fabric2COCO()
train_instance = fabric2coco.to_coco(anno_dir,img_dir)
#if not os.path.exists("coco/annotations/"):
#    os.makedirs("coco/annotations/")
fabric2coco.save_coco_json(train_instance, "F:/deep_learning/dataset/2019tianchi/round1/train/Annotations/train.json")

'''需要注意的是，在标注文件中，给出了具体的疵点名称defect_name，而转换脚本中直接映射为评测中用到的category_id。defect_name到category_id是在训练前映射还是在前传时映射可以自由选择。'''