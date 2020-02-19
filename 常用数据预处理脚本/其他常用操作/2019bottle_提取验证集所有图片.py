#!/usr/bin/env python
# coding=UTF-8


from pycocotools.coco import COCO
import shutil
import os

#img_and_anno_root = '/mfs/home/fangyong/data/guangdong/round1/train/'
#img_and_anno_root ='K:/deep_learning/dataset/2019tianchi/train/'
img_and_anno_root = 'C:/Users/62349/Downloads/chongqing1_round2_train_20200213/'

img_path = img_and_anno_root + 'pingshen/'
annFile = img_and_anno_root + 'val_pingshen.json'

img_save_path = img_and_anno_root + 'val_pingshen'

if not os.path.exists(img_save_path):
    os.makedirs(img_save_path)

# 初始化标注数据的 COCO api
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())

cats = sorted(cats, key = lambda e:e.get('id'),reverse = False) # clw note：这里并不完全是COCO格式，只能算是类COCO格式，因此
                                                                #           比如这里的categories就不是排序的，因此需要手动排序

img_list = os.listdir(img_path)


imgids = coco.getImgIds()
for i, imgId in enumerate(imgids):
    img = coco.loadImgs(imgId)[0]
    image_name = img['file_name']

    print('clw: already read {} images, image_name: {}'.format(i+1, image_name))
    # print(img)

    shutil.copy(os.path.join(img_path, image_name), os.path.join(img_save_path, image_name) )
    #img = cv2.imread(os.path.join(img_path, image_name))
    #cv2.imwrite(os.path.join(img_save_path, image_name), img)

'''
    # #加载并显示图片
    # I = io.imread('%s/%s' % (img_path, img['file_name']))
    # plt.axis('off')
    # plt.imshow(I)
    # plt.show()

    # catIds=[] 说明展示所有类别的box，也可以指定类别

    # clw note:读取image_id对应标注信息的两种方法：
    #           method 1, don't fit to this dataset
    # annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
    # anns = coco.loadAnns(annIds)

    # method 2, fit this dataset!!!
    anns = coco.imgToAnns[imgId]
    #print(anns)

    coordinates = []
    labels = []
    img = cv2.imread(os.path.join(img_path, image_name))

    special_class_num = 0  # clw note: only see some special class
    for j, ann in enumerate(anns):
        # if ann['category_id'] != 4:  # clw note: only see some special class
        #     continue
        #special_class_num += 1
        # 1、求坐标
        coordinate = []
        # coordinate.append(anns[j]['bbox'][0])
        # coordinate.append(anns[j]['bbox'][1]+anns[j]['bbox'][3])
        # coordinate.append(anns[j]['bbox'][0]+anns[j]['bbox'][2])
        # coordinate.append(anns[j]['bbox'][1])
        coordinate.append(anns[j]['bbox'][0])
        coordinate.append(anns[j]['bbox'][1])
        coordinate.append(anns[j]['bbox'][2])
        coordinate.append(anns[j]['bbox'][3])
        # print(coordinate)
        coordinates.append(coordinate)

        # 2、找到对应的标签
        labels.append(cats[anns[j]['category_id'] - 1 ]['name'])  # clw note: start from 1, not 0; the reason to -1 is that the first is 1 but the index should be 0
        #labels.append(cats[anns[j]['category_id']]['name'])  # clw note: start from 0;

        img = draw_rectangle(coordinates, labels, img)

        #if cats[anns[j]['category_id'] - 1]['id']== 13:
        #if cats[anns[j]['category_id'] ]['id'] == 0:
        #    flag_SHOW = True

    #if not flag_SHOW:
    #    continue
    #flag_SHOW = False

    if NEED_SAVE:
        cv2.imwrite(os.path.join(img_save_path, image_name), img)
    cv2.namedWindow(image_name, 0);
    cv2.resizeWindow(image_name, 1600, 1200);
    cv2.moveWindow(image_name, 0, 0);
    cv2.imshow(image_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''