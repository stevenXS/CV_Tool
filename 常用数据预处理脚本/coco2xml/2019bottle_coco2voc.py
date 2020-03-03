# clw note:注意headstr要根据实际情况修改folder,写成训练集或验证集所在文件夹名称
# 如训练集写成train,验证集写成val,后续训练腾讯优图的YOLOv3会用到

from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import time

dataDir = 'C:/Users/62349/Downloads/chongqing1_round1_train_20191223/' # clw modify
# 瓶盖
# img_folder = 'images'
# annotations_list = ['annotations1']
# savepath = os.path.join(dataDir, 'xml1')   # clw note: the path you want to save your results for coco to voc

# 瓶身
img_folder = 'images'
annotations_list = ['annotations_pinggai_processed_modified']
savepath = os.path.join(dataDir, 'xml_pinggai')



# img_dir = os.path.join(savepath, 'images')
# anno_dir = os.path.join(savepath, 'Annotations')
img_dir = os.path.join(savepath, 'dataset')
anno_dir = img_dir

if not os.path.exists(savepath):
    os.makedirs(savepath)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
if not os.path.exists(anno_dir):
    os.makedirs(anno_dir)


datasets_list = ['train']  # clw note:need modify 2

classes_ids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

# clw note: need modify 3(folder)
headstr = """\
<annotation>
    <folder>train</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''

def id2name(coco):
    classes = dict()
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']
    return classes


def write_xml(anno_path, head, objs, tail):
    f = open(anno_path, "w")
    f.write(head)
    for obj in objs:
        f.write(objstr % (obj[0], obj[1], obj[2], obj[3], obj[4]))
    f.write(tail)


def save_annotations_and_imgs(coco, filename, objs):
    # eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path = os.path.join(anno_dir, filename[:-3] + 'xml')
    #img_path = dataDir + dataset + '/' + filename
    img_path = os.path.join(dataDir, img_folder, filename)  # clw modify: 根据具体情况来
    #print(img_path)
    dst_imgpath = os.path.join(img_dir, filename)

    img = cv2.imread(img_path)
    if (img.shape[2] == 1):
        print(filename + " not a RGB image")
        return
    shutil.copy(img_path, dst_imgpath)

    head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path, head, objs, tail)


def showimg(coco, img, classes, cls_id, show=True):
    global dataDir
    I = Image.open(os.path.join(dataDir, img_folder, img['file_name']))
    #I = Image.open('%s/%s/%s' % (dataDir, dataset, img['file_name']))
    #I = Image.open('%s/%s%s%s' % (dataDir, dataset, img_folder, img['file_name']))

    # 通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=int(cls_id), iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        #class_name = classes[ann['category_id']]  # clw modify: 因为xml中最好不要有中文，因此考虑存classes_ids，见下
        #if class_name in classes_names:

        class_name = classes_ids[ann['category_id'] - 1]
        if class_name in classes_ids:
            #print(class_name)
            if 'bbox' in ann:
                bbox = ann['bbox']
                xmin = int(bbox[0])
                ymin = int(bbox[1])
                xmax = int(bbox[2] + bbox[0])
                ymax = int(bbox[3] + bbox[1])
                obj = [class_name, xmin, ymin, xmax, ymax]
                objs.append(obj)
                draw = ImageDraw.Draw(I)
                draw.rectangle([xmin, ymin, xmax, ymax])
    if show:
        plt.figure()
        plt.axis('off')
        plt.imshow(I)
        plt.show()

    return objs


for annotation in annotations_list:
    # ./COCO/annotations/instances_train2014.json
    #annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)
    annFile = os.path.join(dataDir, annotation + '.json')

    # COCO API for initializing annotated data
    coco = COCO(annFile)
    '''
    COCO 对象创建完毕后会输出如下信息:
    loading annotations into memory...
    Done (t=0.81s)
    creating index...
    index created!
    至此, json 脚本解析完毕, 并且将图片和对应的标注数据关联起来.
    '''
    # show all classes in coco
    classes = id2name(coco)
    print(classes)
    print('class_ids:', classes_ids)

    for cls in classes_ids:
        # Get ID number of this class
        img_ids = coco.getImgIds(catIds=int(cls))
        print(cls, len(img_ids))
        time.sleep(0.1)  # clw note：防止和tqdm在控制台打印冲突
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            # print(filename)
            objs = showimg(coco, img, classes, cls, show=False)
            #print(objs)
            save_annotations_and_imgs(coco, filename, objs)