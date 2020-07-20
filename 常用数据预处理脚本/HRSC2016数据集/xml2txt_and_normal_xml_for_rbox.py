'''
数据标注文件为.xml，具体格式如下
<HRSC_Image>
  ...
  <Img_SizeWidth>1172</Img_SizeWidth>
  <Img_SizeHeight>816</Img_SizeHeight>
  <Annotated>1</Annotated>
  <HRSC_Objects>
    <HRSC_Object>
      ...
      <box_xmin>119</box_xmin>
      <box_ymin>75</box_ymin>
      <box_xmax>587</box_xmax>
      <box_ymax>789</box_ymax>
      <mbox_cx>341.2143</mbox_cx>
      <mbox_cy>443.3325</mbox_cy>
      <mbox_w>778.4297</mbox_w>
      <mbox_h>178.2595</mbox_h>
      <mbox_ang>-1.122944</mbox_ang>
      ...
    </HRSC_Object>
  </HRSC_Objects>
</HRSC_Image>
'''

# xml文件的读取方法：
# import xml.etree.ElementTree as ET
# tree = ET.parse(xml_file_path)
# root = tree.getroot()
# for obj in root.findall('object'):
#     label = str(obj.find('name').text)

class_list = ['boat']

import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import math
from utils import get_rotated_coors

root_dir = 'D:/dataset/HRSC2016_dataset/HRSC2016/Train'
# root_dir = 'D:/dataset/HRSC2016_dataset/HRSC2016/Test'
annotation_dir = os.path.join(root_dir, 'Annotations')
image_dir = os.path.join(root_dir, 'AllImages')
annotation_filenames = os.listdir(annotation_dir)
#save_dir = 'D:/dataset/HRSC2016_dataset/HRSC2016/Train/Annotations_txt'
save_dir = image_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
NEED_XML = True

from xml.dom.minidom import Document




for annotation_filename in tqdm(annotation_filenames):
    xml_filepath = os.path.join(annotation_dir, annotation_filename)
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    img_w = int(root.find('Img_SizeWidth').text)
    img_h = int(root.find('Img_SizeHeight').text)
    objs = root.find('HRSC_Objects')

    labels = []
    with open(os.path.join(save_dir, annotation_filename[:-4] + '.txt'), 'w') as f:
        for obj in objs.findall('HRSC_Object'):
            xctr = float(obj.find('mbox_cx').text)
            yctr = float(obj.find('mbox_cy').text)
            w = float(obj.find('mbox_w').text)
            h = float(obj.find('mbox_h').text)
            theta = float(obj.find('mbox_ang').text)
            class_id = 0
            labels.append([xctr, yctr, w, h, theta, class_id])
            bbox_info_str = ""
            # for item in [ class_id, xctr / img_w, yctr / img_h, w / img_w, h / img_h, theta ]:
            #     bbox_info_str += (str(item) + ' ')
            box_8coord = get_rotated_coors([xctr, yctr, w, h, theta])
            box_8coord[0:7:2] /= img_w
            box_8coord[1:8:2] /= img_h
            box = [ class_id ] + list(box_8coord)
            for item in box:
                bbox_info_str += (str(item) + ' ')
            bbox_info_str += '\n'
            f.write(bbox_info_str)

    if NEED_XML:  # xml for the project https://github.com/woshiwwwppp/ryolov3research-pytorch-master 's labelImg tool
        doc = Document()
        annotation = doc.createElement('annotation')
        doc.appendChild(annotation)

        folder = doc.createElement('folder')
        folder_name = doc.createTextNode('HRSC2016')
        folder.appendChild(folder_name)
        annotation.appendChild(folder)

        filename = doc.createElement('filename')
        filename_name = doc.createTextNode(annotation_filename[:-4] + '.jpg') #clw modify
        filename.appendChild(filename_name)
        annotation.appendChild(filename)

        source = doc.createElement('source')
        annotation.appendChild(source)
        database = doc.createElement('database')
        database.appendChild(doc.createTextNode('Unknown'))
        source.appendChild(database)

        size = doc.createElement('size')
        annotation.appendChild(size)
        width = doc.createElement('width')
        width.appendChild(doc.createTextNode(str(img_w)))
        height = doc.createElement('height')
        height.appendChild(doc.createTextNode(str(img_h)))
        depth = doc.createElement('depth')
        depth.appendChild(doc.createTextNode(str(3)))
        size.appendChild(width)
        size.appendChild(height)
        size.appendChild(depth)
        segmented = doc.createElement('segmented')
        segmented.appendChild(doc.createTextNode('0'))
        annotation.appendChild(segmented)

        for i, label in enumerate(labels):
            objects = doc.createElement('object')
            annotation.appendChild(objects)

            pose = doc.createElement('type')
            pose.appendChild(doc.createTextNode('robndbox'))
            objects.appendChild(pose)

            object_name = doc.createElement('name')
            object_name.appendChild(doc.createTextNode(class_list[label[-1]]))
            objects.appendChild(object_name)

            pose = doc.createElement('pose')
            pose.appendChild(doc.createTextNode('Unspecified'))
            objects.appendChild(pose)

            truncated = doc.createElement('truncated')
            truncated.appendChild(doc.createTextNode('0'))
            objects.appendChild(truncated)
            difficult = doc.createElement('difficult')
            difficult.appendChild(doc.createTextNode('0')) # clw note: TODO:后续可以考虑作为参数传进来，但是需要是np.array??仿照box传进来试试
            objects.appendChild(difficult)
            bndbox = doc.createElement('robndbox')
            objects.appendChild(bndbox)

            cx = doc.createElement('cx')
            cx.appendChild(doc.createTextNode(str(label[0])))
            bndbox.appendChild(cx)
            cy = doc.createElement('cy')
            cy.appendChild(doc.createTextNode(str(label[1])))
            bndbox.appendChild(cy)
            w = doc.createElement('w')
            w.appendChild(doc.createTextNode(str(label[2])))
            bndbox.appendChild(w)
            h = doc.createElement('h')
            h.appendChild(doc.createTextNode(str(label[3])))
            bndbox.appendChild(h)
            angle = doc.createElement('angle')
            #aaa = str(-label[4] )
            angle.appendChild(doc.createTextNode(str(label[4] )))  #  / math.pi * 180  弧度转角度
            bndbox.appendChild(angle)

        with open(os.path.join(save_dir, annotation_filename[:-4] + '.xml'), 'w') as f:
            f.write(doc.toprettyxml(indent='\t'))

# if __name__ == '__main__':

