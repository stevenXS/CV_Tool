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

def convert_annotation(txt_path):
    img = cv2.imread(txt_path[:-4] + '.png')
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
    image_names = [i for i in os.listdir(img_path) if i.endswith(".png") or i.endswith(".jpg")]
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
            print('time use: %.3fs' %(time.time()-start) )
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
        all_boxes = self.txt2boxes()  # all gt box in all txts
        result = self.kmeans(all_boxes, k=self.cluster_number)  # clw note: find 9 boxes in n boxes, which has
        result_ratio = result[np.lexsort(result.T[0, None])]  # clw note TODO: 按照第一维度排序，理论上应该按面积排序
        self.result2txt(result)

        nAnchor = len(result_ratio)
        anchor = result_ratio[0]
        format_anchors = str(anchor[0]) + "," + str(anchor[1]) + "," + str(anchor[2] * 180 / math.pi)   # str(anchor[2])
        for i in range(1, nAnchor):
            anchor = result_ratio[i]
            format_anchors += ",  " + str(anchor[0]) + "," + str(anchor[1])+ "," + str(anchor[2]* 180 / math.pi)  # str(anchor[2])

        # print("\nK anchors: {}".format(format_anchors))
        # print("Accuracy: {:.2f}%".format( self.avg_iou(all_boxes, result) * 100))  # clw note
        # pass
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
    # for i in tqdm(range(0, 10)):  # clw modify:多次聚类，比如聚类10次，输出最大的acc和对应的anchor
    #     kmeans = YOLO_Kmeans(cluster_number, filename)
    #     anchors, acc = kmeans.txt2clusters()
    #     if acc > acc_max:
    #         acc_max = acc
    #         anchors_max = anchors
    # print("K anchors: {}".format(anchors_max))
    # print("Accuracy: {:.2f}%".format(acc_max))


if __name__ == "__main__":

    img_path = 'D:/dataset/DOTA/image'  # 暂时认为 img 和 xml 在同一文件夹
    if not os.path.exists(img_path):
        raise Exception("not exists '%s'" % (img_path))

    save_path = 'train_info.txt'
    scan_annotations(img_path, save_path)  # 扫描所有txt，将训练集图片和对应box信息写入txt中
    kmeans_anchors(save_path, 9)  # 读取上面的txt，得到所有bbox，然后做聚类
    pass