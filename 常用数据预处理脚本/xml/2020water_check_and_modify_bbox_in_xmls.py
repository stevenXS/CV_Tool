# written by clw
# 注意这个比赛用的是非标xml，格式和标准的略有区别！
# 检查并修改bbox是否存在越界，顺便统计下每个类别的个数

import xml.etree.ElementTree as ET
import os
import cv2

root_path =  'D:/2020water/train/'
xml_path =  os.path.join(root_path, 'box')
img_path =  os.path.join(root_path, 'image')
class_list = ['holothurian', 'echinus', 'scallop', 'starfish']
img_format = '.jpg'
class_count = {'holothurian':0, 'echinus':0, 'scallop':0, 'starfish':0} # 顺便统计下每个类别的个数

file_names = os.listdir(xml_path)
xml_file_names = []
for file_name in file_names:
    if file_name.endswith('.xml'):
        xml_file_names.append(file_name)


def check_and_modify_bbox_in_all_xmls(xml_path):
    for idx, xml_file_name in enumerate(xml_file_names):
        xml_file_path = os.path.join(xml_path, xml_file_name)
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # 先找到该xml对应的图片，获取h和w
        # img = cv2.imread(os.path.join(img_path, xml_file_name[:-4] + img_format))
        # h, w = img.shape[0], img.shape[1]
        clear_objs = []
        # 遍历该xml里面所有bbox，判断有无越界
        for obj in root.findall('object'):
            label = str(obj.find('name').text)
            if label not in class_list:
                if label == 'waterweeds':
                    clear_objs.append(obj)
                else:
                    assert False, 'warning found a new label：%s' % label
            else:
                class_count[label] += 1
            xmin = int(obj.find('bndbox').find("xmin").text)
            ymin = int(obj.find('bndbox').find("ymin").text)
            xmax = int(obj.find('bndbox').find("xmax").text)
            ymax = int(obj.find('bndbox').find("ymax").text)

            # （1）判断w、h是否小于一定值，比如5，也就是xmax - xmin和ymax - ymin
            if xmax - xmin <= 5 or ymax - ymin <= 5:
                print('idx %d, image %s , w =%d, h=%d' % (idx + 1, xml_file_name[:-4] + img_format, xmax - xmin, ymax-ymin))

            # （2）判断xy是否越界，正常比如1024的图，bbox的x和y应该在1~1024才对
            # if xmin <= 0 or xmin > w:
            #     print('idx %d, image %s has wrong coord, xmin =%d, and image width=%d' % (idx+1, xml_file_name[:-4] + img_format, xmin, w))
            #     if xmin > w:
            #         obj.find('bndbox').find("xmin").text = str(w)
            #     elif xmin <= 0:
            #         obj.find('bndbox').find("xmin").text = str(1)
            #
            # if xmax <= 0 or xmax > w:
            #     print('idx %d, image %s has wrong coord, xmax =%d, and image width=%d' % (idx+1, xml_file_name[:-4] + img_format, xmax, w))
            #     if xmax > w:
            #         obj.find('bndbox').find("xmax").text = str(w)
            #     elif xmax <= 0:
            #         obj.find('bndbox').find("xmax").text = str(1)
            #
            # if ymin <= 0 or ymin > h:
            #     print('idx %d, image %s has wrong coord, ymin =%d, and image height=%d' % (idx+1, xml_file_name[:-4] + img_format, ymin, h))
            #     if ymin > h:
            #         obj.find('bndbox').find("ymin").text = str(h)
            #     elif ymin <= 0:
            #         obj.find('bndbox').find("ymin").text = str(1)
            #
            # if ymax <= 0 or ymax > h:
            #     print('idx %d, image %s has wrong coord, ymax =%d, and image height=%d' % (idx+1, xml_file_name[:-4] + img_format, ymax, h))
            #     if ymax > h:
            #         obj.find('bndbox').find("ymax").text = str(h)
            #     elif ymax <= 0:
            #         obj.find('bndbox').find("ymax").text = str(1)

        # 清掉不需要的类别的obj，如waterweeds
        for obj in clear_objs:
            root.remove(obj)

        # 重写xml
        #tree.write(xml_file_path)

    print(class_count)
    print('end!')

if __name__ == "__main__":
    check_and_modify_bbox_in_all_xmls(xml_path)