#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#55493
from __future__ import print_function
from collections import Counter
from turtle import back
import cv2 as cv
import argparse
import numpy as np
import os
import xml.dom.minidom as xmldom
from nms import py_cpu_nms
from hecheng import frame2video
#import detectron2
#from detectron2.utils.logger import setup_logger
from itertools import product
import time as T
import re
import linecache
import numpy as np

parser = argparse.ArgumentParser(
    description=
    'This program shows how to use background subtraction methods provided by OpenCV. You can process both videos and images.'
)
parser.add_argument('--algo',
                    type=str,
                    help='Background subtraction method (KNN, MOG2).',
                    default='VIBE')
parser.add_argument('--video',
                    type=str,
                    default="/mnt/data2/gaokongdataset/video/")
parser.add_argument('--dataset',
                    type=str,
                    default="/mnt/data2/gaokongdataset/dataset/")
parser.add_argument('--num', type=int, help='.', default='7')
parser.add_argument(
    '--val',
    type=str,
    help='输入视频',
    default="/mnt/data2/gaokongdataset/Annotation/weak_light_test.txt")
parser.add_argument(
    '--test_val',
    type=str,
    help='少量val数据',
    default="/mnt/data2/gaokongdataset/Annotation/little_val.txt")
parser.add_argument('--time',
                    type=str,
                    help='beginning and ending',
                    default="/mnt/data2/gaokongdataset/start.txt")
parser.add_argument(
    '--result_txt',
    type=str,
    help="输出的txt",
    default=
    "/mnt/data1/zzb/gaokongpaowu/IOU_0.3_output/VIBE_weak_light_test.txt")

args = parser.parse_args()


class ViBe:
    '''
    ViBe运动检测，分割背景和前景运动图像
    '''

    def __init__(self, num_sam=20, min_match=2, radiu=20, rand_sam=16):
        self.defaultNbSamples = num_sam  #每个像素的样本集数量，默认20个
        self.defaultReqMatches = min_match  #前景像素匹配数量，如果超过此值，则认为是背景像素
        self.defaultRadius = radiu  #匹配半径，即在该半径内则认为是匹配像素
        self.defaultSubsamplingFactor = rand_sam  #随机数因子，如果检测为背景，每个像素有1/defaultSubsamplingFactor几率更新样本集和领域样本集

        self.background = 0
        self.foreground = 255

    def __buildNeighborArray(self, img):
        '''
        构建一副图像中每个像素的邻域数组
        参数：输入灰度图像
        返回值：每个像素9邻域数组，保存到self.samples中
        '''
        height, width = img.shape
        self.samples = np.zeros((self.defaultNbSamples, height, width),
                                dtype=np.uint8)

        #生成随机偏移数组，用于计算随机选择的邻域坐标
        ramoff_xy = np.random.randint(-1,
                                      2,
                                      size=(2, self.defaultNbSamples, height,
                                            width))
        #ramoff_x=np.random.randint(-1,2,size=(self.defaultNbSamples,2,height,width))

        #xr_=np.zeros((height,width))
        xr_ = np.tile(np.arange(width), (height, 1))
        #yr_=np.zeros((height,width))
        yr_ = np.tile(np.arange(height), (width, 1)).T

        xyr_ = np.zeros((2, self.defaultNbSamples, height, width))
        for i in range(self.defaultNbSamples):
            xyr_[1, i] = xr_
            xyr_[0, i] = yr_

        xyr_ = xyr_ + ramoff_xy

        xyr_[xyr_ < 0] = 0
        tpr_ = xyr_[1, :, :, -1]
        tpr_[tpr_ >= width] = width - 1
        tpb_ = xyr_[0, :, -1, :]
        tpb_[tpb_ >= height] = height - 1
        xyr_[0, :, -1, :] = tpb_
        xyr_[1, :, :, -1] = tpr_

        #xyr=np.transpose(xyr_,(2,3,1,0))
        xyr = xyr_.astype(int)
        self.samples = img[xyr[0, :, :, :], xyr[1, :, :, :]]

    def ProcessFirstFrame(self, img):
        '''
        处理视频的第一帧
        1、初始化每个像素的样本集矩阵
        2、初始化前景矩阵的mask
        3、初始化前景像素的检测次数矩阵
        参数：
        img: 传入的numpy图像素组，要求灰度图像
        返回值：
        每个像素的样本集numpy数组
        '''
        self.__buildNeighborArray(img)
        self.fgCount = np.zeros(img.shape)  #每个像素被检测为前景的次数
        self.fgMask = np.zeros(img.shape)  #保存前景像素

    def Update(self, img):
        '''
        处理每帧视频，更新运动前景，并更新样本集。该函数是本类的主函数
        输入：灰度图像
        '''
        height, width = img.shape
        #计算当前像素值与样本库中值之差小于阀值范围RADIUS的个数，采用numpy的广播方法
        dist = np.abs(
            (self.samples.astype(float) - img.astype(float)).astype(int))
        dist[dist < self.defaultRadius] = 1
        dist[dist >= self.defaultRadius] = 0
        matches = np.sum(dist, axis=0)
        #如果大于匹配数量阀值，则是背景，matches值False,否则为前景，值True
        matches = matches < self.defaultReqMatches
        self.fgMask[matches] = self.foreground
        self.fgMask[~matches] = self.background
        #前景像素计数+1,背景像素的计数设置为0
        self.fgCount[matches] = self.fgCount[matches] + 1
        self.fgCount[~matches] = 0
        #如果某个像素连续50次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点
        fakeFG = self.fgCount > 50
        matches[fakeFG] = False
        #此处是该更新函数的关键
        #更新背景像素的样本集，分两个步骤
        #1、每个背景像素有1/self.defaultSubsamplingFactor几率更新自己的样本集
        ##更新样本集方式为随机选取该像素样本集中的一个元素，更新为当前像素的值
        #2、每个背景像素有1/self.defaultSubsamplingFactor几率更新邻域的样本集
        ##更新邻域样本集方式为随机选取一个邻域点，并在该邻域点的样本集中随机选择一个更新为当前像素值
        #更新自己样本集
        upfactor = np.random.randint(self.defaultSubsamplingFactor,
                                     size=img.shape)  #生成每个像素的更新几率
        upfactor[matches] = 100  #前景像素设置为100,其实可以是任何非零值，表示前景像素不需要更新样本集
        upSelfSamplesInd = np.where(upfactor == 0)  #满足更新自己样本集像素的索引
        upSelfSamplesPosition = np.random.randint(
            self.defaultNbSamples,
            size=upSelfSamplesInd[0].shape)  #生成随机更新自己样本集的的索引
        samInd = (upSelfSamplesPosition, upSelfSamplesInd[0],
                  upSelfSamplesInd[1])
        self.samples[samInd] = img[upSelfSamplesInd]  #更新自己样本集中的一个样本为本次图像中对应像素值

        #更新邻域样本集
        upfactor = np.random.randint(self.defaultSubsamplingFactor,
                                     size=img.shape)  #生成每个像素的更新几率
        upfactor[matches] = 100  #前景像素设置为100,其实可以是任何非零值，表示前景像素不需要更新样本集
        upNbSamplesInd = np.where(upfactor == 0)  #满足更新邻域样本集背景像素的索引
        nbnums = upNbSamplesInd[0].shape[0]
        ramNbOffset = np.random.randint(-1, 2, size=(2, nbnums))  #分别是X和Y坐标的偏移
        nbXY = np.stack(upNbSamplesInd)
        nbXY += ramNbOffset
        nbXY[nbXY < 0] = 0
        nbXY[0, nbXY[0, :] >= height] = height - 1
        nbXY[1, nbXY[1, :] >= width] = width - 1
        nbSPos = np.random.randint(self.defaultNbSamples, size=nbnums)
        nbSamInd = (nbSPos, nbXY[0], nbXY[1])
        self.samples[nbSamInd] = img[upNbSamplesInd]

    def getFGMask(self):
        '''
        返回前景mask
        '''
        return self.fgMask


def calculate(bound, mask):

    x, y, w, h = bound

    area = mask[y:y + h, x:x + w]

    pos = area > 0 + 0

    score = np.sum(pos) / (w * h)

    return score


def nms_cnts(cnts, mask, min_area):

    bounds = [cv.boundingRect(c) for c in cnts if cv.contourArea(c) > min_area]

    if len(bounds) == 0:
        return []

    scores = [calculate(b, mask) for b in bounds]

    bounds = np.array(bounds)

    scores = np.expand_dims(np.array(scores), axis=-1)
    scores = np.array(scores)

    keep = py_cpu_nms(np.hstack([bounds, scores]), 0.3)

    return bounds[keep]


#input formal[xmin, ymin, xmax, ymax]
# boxA是真值
def IOU(boxA, boxB):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def getbounds(fgMask):
    #time_start=time.time()
    line = cv.getStructuringElement(cv.MORPH_RECT, (args.num, args.num),
                                    (-1, -1))  # 返回指定形状和尺寸的结构
    fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, line)

    contours, hierarchy = cv.findContours(fgMask, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    bounds = nms_cnts(contours, fgMask, 5)

    deletelist = []
    final_bounds = []
    n = 0  #存行索引
    #  for b in bounds:
    #      x, y, w, h = b
    #      for x_, y_ in product(range(x, x + w), range(y, y + h)):
    #          if not (x_, y_) in final_list:
    #              deletelist.append(n)
    #              break
    #      if w > 4 * h or w < 4 * h:
    #          deletelist.append(n)
    #      n = n + 1
    #  bounds = np.delete(bounds, deletelist, 0)
    #  bounds[:, 2] = bounds[:, 0] + bounds[:, 2]
    #  bounds[:, 3] = bounds[:, 1] + bounds[:, 3]
    #  for b in bounds:
    #      x, y, w, h = b
    #      if (x, y) in final_list:
    #          final_bounds.append(b)

    final_bounds = np.array(bounds)
    if final_bounds.size > 0:
        final_bounds[:, 2] = final_bounds[:, 0] + final_bounds[:, 2]
        final_bounds[:, 3] = final_bounds[:, 1] + final_bounds[:, 3]


#time_end=time.time()
# print('time cost',time_end-time_start,'s')
    return final_bounds


def precision(zhenshu, line, box):
    global right, RIGHT, wrong, WRONG, iou_num, CHACHU, WEICHACHU
    chachu = 0
    weichachu = 0
    if not os.path.exists(args.dataset + line + '/Annotations/' +
                          "frame{}.xml".format(zhenshu)):
        boxA = [0, 0, 0, 0]
        wrong = len(box)
        WRONG = wrong + WRONG
    else:
        xml_file = xmldom.parse(args.dataset + line + '/Annotations/' +
                                "frame{}.xml".format(zhenshu))
        eles = xml_file.documentElement
        for i in range(len(eles.getElementsByTagName("xmin"))):
            xmin = eles.getElementsByTagName("xmin")[i].firstChild.data
            ymin = eles.getElementsByTagName("ymin")[i].firstChild.data
            xmax = eles.getElementsByTagName("xmax")[i].firstChild.data
            ymax = eles.getElementsByTagName("ymax")[i].firstChild.data
            boxA = [xmin, ymin, xmax, ymax]
            for boxB in box:
                if (len(boxA) and len(boxB) and IOU(boxA, boxB) >= iou_num):
                    right = right + 1
                    RIGHT = RIGHT + 1
                    chachu = chachu + 1
                    CHACHU = CHACHU + 1
                else:
                    wrong = wrong + 1
                    WRONG = WRONG + 1
        weichachu = len(eles.getElementsByTagName("xmin")) - chachu
        WEICHACHU = WEICHACHU + weichachu


iou_num = 0.3
f = open(args.result_txt, 'a')
RIGHT = 0.00
WRONG = 0.00
ALL = 0.00
begin = 0
end = 0
tro = 0
CHACHU = 0
WEICHACHU = 0
TRO = 0
TIME = 0
ZHENSHU = 0

pattern = "video"
with open(args.val) as f1:
    timelist = []
    for line in f1:
        start_time = T.time()
        line = re.sub(pattern, "", line.rstrip())
        input = args.video + "{}.mp4".format(line)
        filename = args.dataset + line

        right = 0.00
        wrong = 0.00
        zhenshu = 0

        capture = cv.VideoCapture(input)
        if capture.isOpened():
            zhenshu = zhenshu + 1
            rval, frame = capture.read()
        else:
            rval = False

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        vibe = ViBe()
        vibe.ProcessFirstFrame(frame)

        while rval:
            rval, frame = capture.read()

            # 将输入转为灰度图
            if frame is None:
                break
            zhenshu = zhenshu + 1
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # 输出二值图
            #(segMat, samples) = update(gray, samples)
            vibe.Update(gray)
            segMat = vibe.getFGMask()
            segMat = segMat.astype(np.uint8)
            bounds = getbounds(segMat)
            if len(bounds) > 0:
                timelist.append(zhenshu)
            precision(zhenshu, line, bounds)
            #　转为uint8类型
            # segMat = segMat.astype(np.uint8)
        if len(timelist) == 0:
            begin = 0
            end = 0
        else:
            end = timelist[-1]
            begin = timelist[0]
        # ipdb.set_trace()
        text = linecache.getline(args.time, int(line))
        time = []
        result = re.finditer(",", text)
        for i in result:
            time.append(i.span()[0])
        num = text[0:time[0]]
        if num != line:
            break
        gt_begin = int(text[time[0] + 1:time[1]])
        gt_end = int(text[time[1] + 1:-1])
        if end == 0:
            tro = 0
        elif begin > gt_end or end < gt_begin:
            tro = 0
        else:
            tro = (min(gt_end, end) - max(gt_begin, begin) +
                   1) / (max(gt_end, end) - min(gt_begin, begin) + 1)

        TRO = TRO + tro
        ALL = ALL + 1
        if (right + wrong) > 0:
            f.write(line + '精度为{:.2f} 正确的:{}  错误的{} tro{}\n'.format(
                right / (right + wrong), right, wrong, tro))
        else:
            f.write(num + '一个未查出精度为0\n')
        print("{}done".format(line))
        end_time = T.time()
        ZHENSHU = ZHENSHU + zhenshu
        TIME = end_time - start_time + TIME
Precision = (RIGHT / (RIGHT + WRONG))
Recall = (CHACHU / (CHACHU + WEICHACHU))
F_measure = (2 * Precision * Recall) / (Precision + Recall)
f.write('算法%s总精度为%f\n' % (args.algo, Precision))
f.write('算法%s查全率为%f\n' % (args.algo, Recall))
f.write('算法%s的tro为%f\n' % (args.algo, (TRO / ALL)))
f.write('算法%s的fps为%f\n' % (args.algo, (ZHENSHU / TIME)))
f.write("算法%s的F-measure为%f\n" % (args.algo, F_measure))