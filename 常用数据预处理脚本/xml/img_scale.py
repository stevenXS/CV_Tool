import xml.etree.ElementTree as ET
import os
import cv2

xml_path = 'C:/Users/62349/Downloads/chongqing1_round1_train1_20191223_split/xml1/all/'
img_path = xml_path

save_scale_root = os.path.join(xml_path, "scale")
if not os.path.exists(save_scale_root):
    os.makedirs(save_scale_root)

file_names = os.listdir(xml_path)

img_file_names = []
for file_name in file_names:
    if file_name.endswith('.jpg') or file_name.endswith('.png'):
        img_file_names.append(file_name)

xml_file_names = []
for file_name in file_names:
    if file_name.endswith('.xml'):
        xml_file_names.append(file_name)

assert len(img_file_names) == len(xml_file_names)
print('clw: img_file_nums:', len(img_file_names))

scale_height = 1024
scale_width = 1024

def modify_the_item_in_all_xmls(xml_path):
    for idx, (img_file_name, xml_file_name) in enumerate(zip(img_file_names, xml_file_names)):
        print('clw: idx = ', idx)

        # 1、图片缩放，保存
        img_file_path = os.path.join(img_path, img_file_name)
        img = cv2.imread(img_file_path)
        img_h = img.shape[0]
        img_w = img.shape[1]
        image_scale(img, img_file_name)

        # 2、坐标变化信息写入xml
        scale_height_factor = img_h / scale_height
        scale_width_factor = img_w / scale_width
        xml_file_path = os.path.join(xml_path, xml_file_name)
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        size = root.find('size')
        size.find('height').text = str(scale_height)
        size.find('width').text = str(scale_width)

        for obj in root.iter('object'):
            bbox = obj.find('bndbox')
            bbox.find('xmin').text = str(float(bbox.find('xmin').text) / scale_width_factor)
            bbox.find('ymin').text = str(float(bbox.find('ymin').text) / scale_height_factor)
            bbox.find('xmax').text = str(float(bbox.find('xmax').text) / scale_width_factor)
            bbox.find('ymax').text = str(float(bbox.find('ymax').text) / scale_height_factor)
        folder = root.find('folder')
        folder.text = 'train'
        tree.write(os.path.join(save_scale_root, xml_file_name))


def image_scale(img, img_file_name):
    scaled_img = cv2.resize(img,(scale_width, scale_height), interpolation=cv2.INTER_AREA)
    scaled_img_path = os.path.join(save_scale_root, img_file_name)
    cv2.imwrite(scaled_img_path, scaled_img)
    #print("save", scaled_img_path)
    return


modify_the_item_in_all_xmls(xml_path)