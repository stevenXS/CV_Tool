# clw note:注意headstr要根据实际情况修改folder,写成训练集或验证集所在文件夹名称
# 如训练集写成train,验证集写成val,后续训练腾讯优图的YOLOv3会用到
# 注意：这个程序会先扫描所有的类别，然后把有这个类别的图片id写到一个list，即img_ids = coco.getImgIds(catIds=cls_id)
# 这样做可能会有重复，比如类1和类18这两种缺陷都在同一张图，就会写2次xml。可以考虑改成直接把coco转xml

from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw


# the path you want to save your results for coco to voc
savepath = "F:/deep_learning/dataset/2019tianchi/round1/train/"  # clw note: need modify 1
img_dir = savepath + 'img_and_xmls/'
anno_dir = savepath + 'img_and_xmls/'

# if the dir is not exists,make it,else delete it
def mkr(path):
    # if os.path.exists(path):
    #     shutil.rmtree(path)
    #     os.mkdir(path)
    # else:
    #     os.mkdir(path)

    if not os.path.exists(path):
        os.mkdir(path)

mkr(img_dir)
mkr(anno_dir)





#classes_names = ['car', 'bicycle', 'person', 'motorcycle', 'bus', 'truck']
# classes_names = ['破洞', '水渍', '油渍', '污渍', '三丝', '结头', '花板跳', '百脚', '毛粒',
#     '粗经', '松经', '断经', '吊经', '粗维', '纬缩', '浆斑', '整经结', '星跳', '跳花',
#     '断氨纶', '稀密档', '浪纹档', '色差档', '磨痕', '轧痕', '修痕', '烧毛痕', '死皱', '云织',
#     '双纬', '双经', '跳纱', '筘路', '纬纱不良']
classes_names = ['podong', 'shuizi', 'youzi', 'wuzi', 'sansi', 'jietou', 'huabantiao', 'baijiao', 'maoli',
    'cujing', 'songjing', 'duanjing', 'diaojing', 'cuwei', 'weisuo', 'jiangban', 'zhengjingjie', 'xingtiao', 'tiaohua',
    'duananlun', 'ximidang', 'langwendang', 'sechadang', 'mohen', 'yahen', 'xiuhen', 'shaomaohen', 'sizhou', 'yunzhi',
    'shuangwei', 'shuangjing', 'tiaosha', 'koulu', 'weishabuliang']
#classes_names = [1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13, 14, 15,  16, 16, 17, 18, 18, 18,  19,  19,  19, 19, 20, 20, 20, 20,  20, 20, 20]


# classes_names =   ['破洞', '水渍/油渍/污渍', '三丝', '结头', '花板跳', '百脚', '毛粒',
#     '粗经', '松经', '断经', '吊经', '粗维', '纬缩', '浆斑', '整经结', '星跳/跳花',
#     '断氨纶', '稀密档/浪纹档/色差档', '磨痕/轧痕/修痕/烧毛痕', '死皱/云织/双纬/双经/跳纱/筘路/纬纱不良']
# classes_names = ['1', '2', '3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']

# datasets_list=['train2014', 'val2014']
#datasets_list = ['train2017']
datasets_list = ['train']  # clw note:need modify 2
# Store annotations and train2014/val2014/... in this folder
#dataDir = 'E:/datasets/COCO/'
dataDir = 'F:/deep_learning/dataset/2019tianchi/round1/' # clw note：在该目录下，依次进入上面datasets_list里面列出的文件夹，
                                                         # 在文件夹下的/defect_Images目录里面搜图片

# clw note: need modify 3(folder)
# headstr = """\
# <annotation>
#     <folder>train</folder>
#     <filename>%s</filename>
#     <source>
#         <database>My Database</database>
#         <annotation>COCO</annotation>
#         <image>flickr</image>
#         <flickrid>NULL</flickrid>
#     </source>
#     <owner>
#         <flickrid>NULL</flickrid>
#         <name>company</name>
#     </owner>
#     <size>
#         <width>%d</width>
#         <height>%d</height>
#         <depth>%d</depth>
#     </size>
#     <segmented>0</segmented>
# """
headstr = """\
<annotation>
    <folder>train</folder>
    <filename>%s</filename>
    <path>%s</path>
    <source>
        <database>My Database</database>
    </source>
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


def save_annotations_and_imgs(coco, dataset, filename, objs):
    # eg:COCO_train2014_000000196610.jpg-->COCO_train2014_000000196610.xml
    anno_path = anno_dir + filename[:-3] + 'xml'
    #img_path = dataDir + dataset + '/' + filename
    img_path = dataDir + dataset + '/defect_Images/' + filename  # clw modify: 根据具体情况来
    print(img_path)
    dst_imgpath = img_dir + filename

    img = cv2.imread(img_path)
    if (img.shape[2] == 1):
        print(filename + " not a RGB image")
        return
    #shutil.copy(img_path, dst_imgpath)   # clw note：节约时间不复制图片，之后手动复制即可；

    # head = headstr % (filename, img.shape[1], img.shape[0], img.shape[2])
    head = headstr % (filename, img_path, img.shape[1], img.shape[0], img.shape[2])
    tail = tailstr
    write_xml(anno_path, head, objs, tail)


def showimg(coco, dataset, img, classes, cls_id, show=True):
    global dataDir
    #I = Image.open('%s/%s/%s' % (dataDir, dataset, img['file_name']))
    I = Image.open('%s/%s/defect_Images/%s' % (dataDir, dataset, img['file_name']))

    # 通过id，得到注释的信息
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=cls_id, iscrowd=None)
    # print(annIds)
    anns = coco.loadAnns(annIds)
    # print(anns)
    # coco.showAnns(anns)
    objs = []
    for ann in anns:
        class_name = classes[ann['category_id']]
        assert class_name in classes_names
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2] + bbox[0])
            ymax = int(bbox[3] + bbox[1])
            obj = [class_name, xmin, ymin, xmax, ymax]
            objs.append(obj)
            # draw = ImageDraw.Draw(I)
            # draw.rectangle([xmin, ymin, xmax, ymax])
    # if show:
    #     plt.figure()
    #     plt.axis('off')
    #     plt.imshow(I)
    #     plt.show()

    return objs


for dataset in datasets_list:
    # ./COCO/annotations/instances_train2014.json
    #annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataset)
    annFile = '{}/{}/Annotations/{}.json'.format(dataDir, dataset, dataset)

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
    # [1, 2, 3, 4, 6, 8]

    classes_ids = coco.getCatIds(catNms=classes_names)
    #classes_ids = classes_names    # clw note :相当于一个索引的映射,如35类映射到21类,
                                   # 要根据具体情况,比如数据集里面就已经是21类,那么就无需映射了

    print(classes_ids)
    for cls in classes_names:
        # Get ID number of this class
        cls_id = coco.getCatIds(catNms=[cls])
        img_ids = coco.getImgIds(catIds=cls_id)
        print(cls, len(img_ids))
        # imgIds=img_ids[0:10]
        for imgId in tqdm(img_ids):
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            # print(filename)
            objs = showimg(coco, dataset, img, classes, classes_ids, show=False)
            print(objs)
            save_annotations_and_imgs(coco, dataset, filename, objs)