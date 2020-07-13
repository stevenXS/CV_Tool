import os
import xml.dom.minidom
import cv2

# ImgPath = 'C:/Users/Administrator/Desktop/GC1/train_crop/images/'
# AnnoPath = 'C:/Users/Administrator/Desktop/GC1/train_crop/Annotations/'  # xml文件地址
#ImgPath = 'C:/Users/62349/Downloads/chongqing1_round2_train_20200213/jiuye_out5/'
ImgPath = 'C:/Users/62349/Desktop/SmallObjectAugmentation/save/'
AnnoPath = ImgPath

save_path = './'


def draw_anchor(ImgPath, AnnoPath):
    imagelist = os.listdir(ImgPath)
    imagelist = [img for img in imagelist if img.endswith('.jpg') or img.endswith('.png')]
    for image in imagelist:

        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + image
        xmlfile = AnnoPath + image_pre + '.xml'
        # print(image)
        # 打开xml文档
        DOMTree = xml.dom.minidom.parse(xmlfile)
        # 得到文档元素对象
        collection = DOMTree.documentElement
        # 读取图片
        img = cv2.imread(imgfile)

        filenamelist = collection.getElementsByTagName("filename")
        filename = filenamelist[0].childNodes[0].data
        print(filename)
        # 得到标签名为object的信息
        objectlist = collection.getElementsByTagName("object")

        for objects in objectlist:
            # 每个object中得到子标签名为name的信息
            namelist = objects.getElementsByTagName('name')
            # 通过此语句得到具体的某个name的值
            objectname = namelist[0].childNodes[0].data

            bndbox = objects.getElementsByTagName('bndbox')
            # print(bndbox)
            for box in bndbox:
                x1_list = box.getElementsByTagName('xmin')
                x1 = float(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = float(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')  # 注意坐标，看是否需要转换
                x2 = float(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = float(y2_list[0].childNodes[0].data)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=4)
                cv2.putText(img, objectname, (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0),
                           thickness=2)

        # cv2.imwrite(image, img)
        cv2.namedWindow("coarse", 0)
        cv2.resizeWindow("coarse", 800, 800);
        cv2.imshow('coarse', img)
        k = cv2.waitKey(0)  # waitkey代表读取键盘的输入，括号里的数字代表等待多长时间，单位ms。 0代表一直等待
        if k == 27:  # 键盘上Esc键的键值
            cv2.destroyAllWindows()

        cv2.imwrite(save_path + '/' + filename, img)  # save picture

draw_anchor(ImgPath, AnnoPath)