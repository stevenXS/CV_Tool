# coding:utf-8
# 功能：对 txt格式的数据集（训练集）进行聚类，找到合适的 anchor
# 输入：img_path
# 输出：save_path (no need to modify, default: 'train_info.txt')

import os
import sys
import numpy as np
from tqdm import tqdm
import argparse
import cv2
from rbox_iou_np import skew_bbox_iou
import time
import math

#img_format = '.bmp'
img_format = '.jpg'

def convert_annotation(txt_path):
    img = cv2.imread(txt_path[:-4] + img_format)
    img_h, img_w = img.shape[:2]
    with open(txt_path, 'r') as f:
        annotations = ""
        for line in f.readlines():
            box_info = line.strip().split(' ')

            cls = box_info[0]
            # cls = cls.replace(' ', '_')  # 注意这一句非常关键,因为后面会按照空格提取txt内容,如果class name带空格,那么就会有bug
            xctr = float(box_info[1]) * img_w
            yctr = float(box_info[2]) * img_h
            w = float(box_info[3]) * img_w
            h = float(box_info[4]) * img_h
            xmin = xctr - w / 2
            ymin = yctr - h / 2
            xmax = xctr + w / 2
            ymax = yctr + h / 2
            theta = float(box_info[5])  # clw added

            annotations += " " + ",".join([str(a) for a in [xmin, ymin, xmax, ymax, theta]]) + ',' + str(cls)
        return annotations


def scan_annotations(img_path, save_path="train_info.txt"):
    image_names = [i for i in os.listdir(img_path) if i.endswith(img_format)]
    with open(save_path, 'w') as f:
        pbar = tqdm(image_names)
        for image_name in pbar:
            pbar.set_description("Processing %s" % image_name)
            txt_path = os.path.join(img_path, image_name[:-4] + '.txt')  # 暂时认为 img 和 xml 在同一文件夹
            content = os.path.join(img_path, image_name) + convert_annotation(txt_path) + '\n'
            f.write(content)



box_width_min = 999999
box_height_min = 999999
box_width_max = -1
box_height_max = -1
box_width_list = []
box_height_list = []
box_scale_list = []  # clw note:宽高比


class YOLO_Kmeans:
    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters； 另外这里相当于 wh_iou()
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        #accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        accuracy = np.mean([np.max(skew_bbox_iou(boxes, clusters, wh_iou=True), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):  # clw note: np.median，求中位数
        box_number = boxes.shape[0]
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(box_number, k, replace=False)]  # init k clusters，随机从所有box里面选9个，开始聚类
        while True:
            start = time.time()
            #distances = 1 - self.iou(boxes, clusters)
            distances = 1 - skew_bbox_iou(boxes, clusters, wh_iou=True)
            print('One time cluster, time use: %.3fs' %(time.time()-start) )
            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist( boxes[current_nearest == cluster], axis=0)  # update clusters
            last_nearest = current_nearest
        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        with open(self.filename, 'r') as f:
            dataSet = []
            for line in f:
                infos = line.split(" ")
                # 比如C:/Users/Administrator/Desktop/dataset_steer/JPEGImages/0009496A.jpg 407,671,526,771,0 378,757,502,855,0
                # 对应length=3
                length = len(infos)
                for i in range(1, length):  # clw note：这里要从1开始，因为0是图片路径字符串
                    xmax = float(infos[i].split(',')[2])
                    xmin = float(infos[i].split(',')[0])
                    width = xmax - xmin
                    height = float(infos[i].split(',')[3]) - float(infos[i].split(',')[1])
                    theta = float(infos[i].split(',')[4])
                    dataSet.append([width, height, theta])

                    # --------------------------------------------------------------------------
                    # clw add: 统计所有box宽和高的最大最小值
                    global box_width_min
                    global box_height_min
                    global box_width_max
                    global box_height_max

                    if width < box_width_min:
                        box_width_min = width
                    if height < box_height_min:
                        box_height_min = height
                    if width > box_width_max:
                        box_width_max = width
                    if height > box_height_max:
                        box_height_max = height

                    box_width_list.append(width)
                    box_height_list.append(height)
                    box_scale_list.append(round(width / height, 2))
                    # --------------------------------------------------------------------------

            result = np.array(dataSet)
            return result

    def txt2clusters(self):
        start = time.time()
        all_boxes = self.txt2boxes()  # all gt box in all txts
        result = self.kmeans(all_boxes, k=self.cluster_number)  # clw note: find 9 boxes in n boxes, which has
        result_ratio = result[np.lexsort(result.T[0, None])]  # clw note TODO: 按照第一维度排序，理论上应该按面积排序
        self.result2txt(result)

        nAnchor = len(result_ratio)
        anchor = result_ratio[0]
        format_anchors = str(anchor[0]) + "," + str(anchor[1]) + "," + str(anchor[2])   #str(anchor[2] * 180 / math.pi)   # str(anchor[2])
        for i in range(1, nAnchor):
            anchor = result_ratio[i]
            format_anchors += ",  " + str(anchor[0]) + "," + str(anchor[1])+ "," +  str(anchor[2])  #str(anchor[2]* 180 / math.pi)  # str(anchor[2])

        # print("\nK anchors: {}".format(format_anchors))
        # print("Accuracy: {:.2f}%".format( self.avg_iou(all_boxes, result) * 100))  # clw note
        # pass
        print('time use: %.3fs' % (time.time() - start))
        return format_anchors, self.avg_iou(all_boxes, result) * 100  # clw modify


def kmeans_anchors(filename, cluster_number):
    kmeans = YOLO_Kmeans(cluster_number, filename)
    anchors_max, acc_max = kmeans.txt2clusters()
    print("K anchors: {}".format(anchors_max))
    print("Accuracy: {:.2f}%".format(acc_max))  # clw note
    # print('clw: box_width_min = ', box_width_min)
    # print('clw: box_width_max = ', box_width_max)
    # print('clw: box_height_min = ', box_height_min)
    # print('clw: box_height_max = ', box_height_max)
    # print('clw: box_width_list = ', sorted(box_width_list))
    # print('clw: box_height_list = ', sorted(box_height_list))

    ### option : Multiple times kmeans to get better acc, add by clw
    # print('Multiple times kmeans to get better acc:')
    # for i in tqdm(range(0, 4)):  # clw modify:多次聚类，比如聚类10次，输出最大的acc和对应的anchor
    #     print('================ epoch:', i+1)
    #     kmeans = YOLO_Kmeans(cluster_number, filename)
    #     anchors, acc = kmeans.txt2clusters()
    #     if acc > acc_max:
    #         acc_max = acc
    #         anchors_max = anchors
    # print("K anchors: {}".format(anchors_max))
    # print("Accuracy: {:.2f}%".format(acc_max))


if __name__ == "__main__":

    #img_path = 'D:/dataset/DOTA/image'  # 暂时认为 img 和 xml 在同一文件夹
    #img_path = 'D:/dataset/HRSC2016_dataset/HRSC2016/Train/AllImages'
    img_path = 'D:/dataset/hrsc2016/train'
    if not os.path.exists(img_path):
        raise Exception("not exists '%s'" % (img_path))

    save_path = 'train_info.txt'
    scan_annotations(img_path, save_path)  # 扫描所有txt，将训练集图片和对应box信息写入txt中
    kmeans_anchors(save_path, 9)  # 读取上面的txt，得到所有bbox，然后做聚类

    ### HRSC2016 聚类结果：（原始尺寸）
    # K anchors: 153.64470000000006,29.728680000000054,-57.726013521446056,  155.19675,29.394334999999984,40.098182002066956,  303.57180000000005,41.02582000000001,81.26501050614421,  319.33545000000004,44.92217000000002,7.6706758823311665,  349.01929999999993,52.78947000000005,-40.50205995185467,  364.179,58.62256000000002,33.901563233635784,  397.41065000000003,66.58615,62.141971135857844,  398.16859999999997,69.25216999999998,-69.58910466962975,  535.7231000000002,88.19912,-13.074616135565773  Accuracy: 51%
    # K anchors: 134.16639999999995,29.157140000000027,0.1908657,  143.9851500000001,29.08942000000002,-0.99382035,  322.0242999999999,45.81320999999997,1.307876,  350.22569999999996,50.478650000000016,0.653704,  352.1608,50.480260000000015,0.16709065,  354.24030000000005,50.90028000000004,-0.31161045,  354.81139999999994,58.26080000000002,-1.2549245,  356.0322,55.23478,-0.81037565,  635.0943,160.76770000000005,-0.4397524  Accuracy: 49.48%
    # K anchors: 153.60285000000002,29.688135000000017,-1.013242,  155.19675,29.394334999999984,0.6998453,  303.57180000000005,41.02582000000001,1.418342,  319.33545000000004,44.92217000000002,0.13387854999999999,  349.01929999999993,52.797129999999925,-0.7143881,  364.179,58.62256000000002,0.5916939,  397.41065000000003,66.58615,1.084582,  398.8361,69.37175500000006,-1.2166975,  533.6992,87.04564000000005,-0.2296711  Accuracy: 51.07%
    # K anchors: 146.08159999999998, 29.14616000000001, -1.031564, 153.24849999999992, 29.746440000000007, 0.1205965, 298.4135, 40.40505000000002, 1.398906, 319.3420499999999, 43.87968499999994, 0.61844825, 342.31305000000003, 51.46401499999996, -0.8035791000000001, 355.5137, 51.19388999999997, 0.17092765, 399.54335000000003, 69.70184999999998, -1.224561, 420.4684, 69.64202999999998, 1.066791, 440.3544999999999, 73.31273500000003, -0.35417615  Accuracy: 51.11 %

    ### HRSC2016 聚类结果：（resize到608x608）
    # K anchors: 85.39699505541348,22.482303458282956,-0.9972177,  85.4487815181518,22.309772127139354,1.105451,  85.6279806253456,23.668458355631856,0.09985000999999999,  183.51976346644005,38.623342671480145,-1.218114,  189.0705262012693,38.418077033374516,0.626662,  189.53320248007088,39.30565382781458,0.1563291,  190.48912077701107,40.58675092556226,-0.57682135,  191.35439400749064,40.12343195767198,1.295342,  330.95738882578183,102.08168055993517,-0.52013795  Accuracy: 54.11%
    # K anchors: 65.80976227861748, 21.916235726653298, 0.12316565, 101.20643716608595, 24.801213880794705, -1.174248, 148.4480568047337, 28.662507826086994, 0.6268285, 169.8441158734424, 33.518423433623354, 1.2899615, 179.68553283626318, 37.44407146075452, -0.73150325, 181.9370548871432, 37.02365686366138, 0.15708139999999998, 262.797747350715, 58.420735298416616, 0.7094149, 265.19500593723484, 62.20253203883496, -1.196715, 281.43522323624484, 66.83933123876199, -0.2824789  Accuracy: 54.30 %
    ### 改为18个anchor：K anchors: 34.37395545863661,18.7533146677755,-0.096882795,  85.43945888355026,22.234035418868984,0.64415975,  87.27248386457902,22.934356161495884,-1.05316,  90.18134386317911,25.20662566091454,0.080508365,  91.44434032395566,23.359433341375166,1.211161,  159.90682138960898,30.398773777724557,1.4270995,  166.66744069264072,31.68249848159553,-1.2107215,  168.06643110173852,32.701739661537914,0.19420294999999999,  170.353525,32.843392549019654,0.6146669,  172.09315857113165,35.727610711240004,-0.39064135,  193.4506464864865,40.01605647058824,-0.8169922,  195.1503213114754,41.223980652962496,1.057615,  211.8379509012876,49.36749342891278,0.09622911,  220.38822821833162,51.273914602409604,-1.367477,  275.6920569636046,62.18695368647974,0.54654145,  284.26396971502965,65.50240666033736,1.2829245,  300.9885991158267,71.69895205811133,-0.4544155,  331.31774477485135,91.15590329670329,-1.170975  Accuracy: 64.77%
    ### 改为27个anchor：K anchors: 73.7137977215703,22.407025209512312,0.0748906,  77.91302183600817,21.498227030871718,-1.0836425,  85.40576000000001,22.36573544457974,1.13529,  127.92421420017106,29.577465435225953,-0.48633364999999995,  128.4539735267452,25.64585454545454,0.613651,  157.69659198635975,30.07026133333335,1.418342,  164.597433557047,33.157171294697946,-0.9732121,  164.91051855502477,32.083611924323094,0.1701482,  167.14538420878597,39.475073713558345,-0.68268635,  168.30583838383842,32.58734752995497,-1.2265505,  176.47806742502587,36.47190975688818,0.6161824,  181.15991176259655,35.48186527149559,1.1246345,  184.57085733558182,31.54090615384615,-0.8239819,  187.96947096774198,40.02238744241494,-0.16269825,  194.22886611570252,46.878859476961395,0.2359145,  220.4422308421018,50.13387350470117,0.9354458,  228.68667554709805,51.991558716049326,1.41733,  228.79395700680277,52.8627505671642,-0.8076533,  252.05747841105352,57.993447770859234,-1.218524,  257.0740294212333,58.89003162244221,-0.45325155,  265.27987832699614,61.38726207228916,0.1457163,  268.12434844450553,58.254391067961194,0.5484803,  280.84501164021157,59.97776444444446,1.195692,  310.2228439932318,101.32781493975904,-0.6703775,  343.33544698179,116.63176800291251,1.0371700000000001,  346.6634316498316,120.13515271411336,-1.243804,  362.97864461670974,123.15015742246312,0.03422084  Accuracy: 68.44%