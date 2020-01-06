# https://github.com/JialianW/pascal2coco/blob/e136f0b8d50b78e8715d2ada89397ea9b7729ef4/pascal2coco.py

import json
import xml.etree.ElementTree as ET
import os
import cv2

###### 需要配置以下2项：
#my_classes = ['1','2','3','4','5','6','7','8','9','10', '11']    #clw note：第二项，添加自己的class，比如person,dog,car等等；目前不知多分类是否可以
my_classes = ['head']

dataset_type = 'train'

#root_path = '/media/clwclw/data/2018yuncong/Part_AB'
root_path = '/media/clwclw/data/2020peopledensity/background'

#annotation_path = os.path.join(root_path, dataset_type) #clw note：所有xml保存文件路径
annotation_path = root_path
image_path = root_path
save_json_path = './train.json' #clw note:保存json的文件路径


def load_load_image_labels(LABEL_PATH, class_name=[]):

    images=[]
    annotations=[]

    #assign your categories which contain the classname and class id
    #the order must be same as the class_name
    categories = []
    for i in range(len(my_classes)):
        #category = dict(id=i+1, name=my_classes[i], supercategory="none")
        category = dict(id=i+1, name=my_classes[i], supercategory="none")
        categories.append(category)

    # load ground-truth from xml annotations
    id_number=0

    xml_files = [img_file for img_file in os.listdir(LABEL_PATH) if img_file.endswith('.xml')]
    # 新增功能:如果xml为空,说明都是背景图片,那么直接写一个只有images和categories,annotations为空的json
    if len(xml_files) == 0:
        print('clw: xml为空,说明都是背景图片,那么直接写一个只有images和categories,annotations为空的json')
        img_names = os.listdir(image_path)
        for image_id, img_name in enumerate(img_names):
            print('image_id:', image_id)
            img = cv2.imread(os.path.join(image_path, img_name))
            images.append({
                "file_name": img_name,
                "height": img.shape[0],
                "width": img.shape[1],
                "id": image_id
            })# id of the image. referenced in the annotation "image_id"


    else:
        for image_id, label_file_name in enumerate(xml_files):
            print('image', str(image_id+1)+':'+label_file_name)
            label_file=os.path.join(LABEL_PATH, label_file_name)
            image_file = label_file_name.split('.')[0] + '.jpg'   # or label_file_name[:-4] + '.jpg'
            tree = ET.parse(label_file)
            root = tree.getroot()

            size=root.find('size')
            width = float(size.find('width').text)
            height = float(size.find('height').text)
            images.append({
                "file_name": image_file,
                "height": height,
                "width": width,
                "id": image_id
            })# id of the image. referenced in the annotation "image_id"

            for anno_id, obj in enumerate(root.iter('object')):
                name = obj.find('name').text
                # if name == '0':  # 去掉背景
                #      continue
                bbox=obj.find('bndbox')
                #cls_id = int(name) + 1  # 0~10改为1~11,以支持mmdetection
                cls_id = 1
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                width = xmax-xmin
                height = ymax-ymin

                assert width > 0, 'xlen即bbox宽度不能小于0,该图片名字为:%s, 对应的xmin, ymin为%d, %d' %(image_file, xmin, ymin)
                assert height > 0, 'ylen即bbox高度不能小于0,该图片名字为:%s,对应的xmin, ymin为%d, %d'  %(image_file, xmin, ymin)

                annotations.append({
                                    #"segmentation" : [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin],],
                                    "segmentation": [],
                                    "area" : width * height,
                                    "iscrowd": 0,
                                    "image_id": image_id,
                                    "bbox" : [xmin, ymin, width, height],  # clw note:根据实际情况修改
                                    "category_id": cls_id,
                                    "id": id_number,   #初始值id_number为0，每处理完一个xml文件，id_number+1
                                    "ignore":0
                                    })
                # print([image_file,image_id, cls_id, xmin, ymin, xlen, ylen])
                id_number += 1

    return {"images":images,"annotations":annotations,"categories":categories}

if __name__=='__main__':
    LABEL_PATH =  annotation_path
    classes    =  my_classes

    label_dict = load_load_image_labels(LABEL_PATH,classes)

    # with open(jsonfile,'w') as json_file:
    #     json_file.write(json.dumps(label_dict, ensure_ascii=False))
    #     json_file.close()
    with open(save_json_path, 'w', encoding='utf-8') as f:
        json.dump(label_dict, f, indent=1, separators=(',', ': '))  # clw note: 用notepad的json格式打开看,结构更清晰
