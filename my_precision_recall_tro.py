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
# import ipdb
import linecache

#22253

parser = argparse.ArgumentParser(
    description=
    'This program shows how to use background subtraction methods provided by OpenCV. You can process both videos and images.'
)
parser.add_argument('--algo',
                    type=str,
                    help='Background subtraction method (KNN, MOG2).',
                    default='GSOC')

parser.add_argument('--num', type=int, help='.', default='7')

parser.add_argument('--video',
                    type=str,
                    default="/mnt/data2/gaokongdataset/video/")
parser.add_argument('--dataset',
                    type=str,
                    default="/mnt/data2/gaokongdataset/dataset/")
parser.add_argument(
    '--val',
    type=str,
    help='输入视频',
    default="/mnt/data2/gaokongdataset/Annotation/resolution_test.txt")
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
    "/mnt/data1/zzb/gaokongpaowu/output/zzb_tro_0.3_GSOC_resolution__300_100_7.txt"
)
# rec整个矩形做了筛选 point只筛选了左上角点
'''parser.add_argument('--output_frame',type=str,help="输出帧",default="/mnt/data1/zzb/gaokongpaowu/output_6_3_ps/")
parser.add_argument('--output_video',type=str,help="输出视频",default="/mnt/data1/zzb/gaokongpaowu/output_6_3_ps/")
'''

args = parser.parse_args()


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


def getbounds(image):
    #time_start=time.time()
    fgMask = backSub.apply(image)
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

    Pos = os.path.splitext(filename)[0].index('_')
    clas = os.path.splitext(filename)[0][0:Pos]
    num = os.path.splitext(filename)[0]
    gtframe = {}
    gtframe['video'] = [num]
    for frame in os.listdir(args.bounding_box + clas + '/' + num + '/' +
                            'Annotations'):
        gtframe_list = os.path.splitext(frame)[0]
        xml_file = xmldom.parse(args.bounding_box + clas + '/' + num + '/' +
                                'Annotations' + '/' + frame)
        eles = xml_file.documentElement
        xmin = eles.getElementsByTagName("xmin")[0].firstChild.data
        ymin = eles.getElementsByTagName("ymin")[0].firstChild.data
        xmax = eles.getElementsByTagName("xmax")[0].firstChild.data
        ymax = eles.getElementsByTagName("ymax")[0].firstChild.data
        gtframe[gtframe_list] = [xmin, ymin, xmax, ymax]
    return gtframe, num


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

        if args.algo == 'MOG2':
            backSub = cv.createBackgroundSubtractorMOG2(300, 100, False)
            # backSub= cv.createBackgroundSubtractorKNN(detectShadows=True)
        elif args.algo == 'GSOC':
            backSub = cv.bgsegm.createBackgroundSubtractorGSOC(300, 100)
        elif args.algo == 'LSBP':
            backSub = cv.bgsegm.createBackgroundSubtractorLSBP()
        elif args.algo == 'GMG':
            backSub = cv.bgsegm.createBackgroundSubtractorGMG()
            backSub.setNumFrames(5)
            backSub.setUpdateBackgroundModel(True)
        elif args.algo == 'CNT':
            backSub = cv.bgsegm.createBackgroundSubtractorCNT()
            backSub.setIsParallel(True)
            backSub.setUseHistory(True)
            backSub.setMinPixelStability(1)
            backSub.setMaxPixelStability(4)
        elif args.algo == 'MOG':
            backSub = cv.bgsegm.createBackgroundSubtractorMOG()
        else:
            backSub = cv.createBackgroundSubtractorKNN(300, 100, False)
        # input = args.input_video + filename
        # capture = cv.VideoCapture(input)  # 用capture.read得到一个视频的frame会一直往后，除非重新用capture = cv.VideoCapture(input)读一次视频
        # fps = capture.get(cv.CAP_PROP_FPS)  # 获得码率及尺寸
        # size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
        #         int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
        # fNUMS = capture.get(cv.CAP_PROP_FRAME_COUNT)
        # if not capture.isOpened:
        #     print('Unable to open: ' + input)
        #     exit(0)

        # 读ground truth帧
        # gtframe, num = getgtframe(filename)

        # 提取建筑物背景
    #  final_list = getfinal_list(capture)

        capture = cv.VideoCapture(input)
        # cv.waitKey(10)
        while True:
            zhenshu = zhenshu + 1
            ret, frame = capture.read()  # ret是bool参数判断有无获取成功，frame返回帧
            if frame is None:
                break

            bounds = getbounds(frame)
            if len(bounds) > 0:
                timelist.append(zhenshu)
            precision(zhenshu, line, bounds)

    #  im_dir = args.output + 'frame/' + num + '/'  # 帧存放路径
    #  #video_dir = "/mnt/data1/zzb/zzb/gaokongpaowu/test_result/" + filename # 合成视频存放的路径
    #  fps = 5  # 帧率，每秒钟帧数越多，所显示的动作就会越流畅
    #  if not os.path.exists(args.output + 'video/' + num):
    #      os.makedirs(args.output + 'video/' + num)
    #  frame2video(im_dir, args.output + 'video/' + filename, fps)

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
