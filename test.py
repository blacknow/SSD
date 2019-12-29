"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import SIXray_ROOT, SIXrayAnnotationTransform, SIXrayDetection, BaseTransform
from data import SIXray_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import os.path as osp
import time
import argparse
import numpy as np
import pickle
import cv2
import shutil

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "a1")


EPOCH = 5
# GPUID = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default="", type=str,
                    help='Trained state_dict file path to open')
parser.add_argument(  # '--save_folder', default='/media/dsg3/husheng/eval/', type=str,
    '--save_folder',
    default="", type=str,
    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float, # 此处0.2改为0.01
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--SIXray_root', default=SIXray_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--imagesetfile',
                    # default='/media/dsg3/datasets/SIXray/dataset-test.txt', type=str,
                    default="D:/2019_fall_semester/01_MachineLearning/ssd/data/VOC_0712/VOCdevkit/core_coreless_test/core_coreless_test.txt",
                    type=str,
                    help='imageset file path to open')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        # print("WARNING: It looks like you have a CUDA device, but aren't using \
        #         CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

### 这里的xml可能需要改动
annopath = os.path.join(args.SIXray_root, 'Anno_test', '%s.xml')
# annopath = os.path.join(args.SIXray_root, 'Anno_core_coreless_battery_sub_2000_500') + os.sep + '%s.xml'
imgpath = os.path.join(args.SIXray_root, 'Image_test', '%s.jpg')
# imgpath = os.path.join(args.SIXray_root, 'cut_Image_core_coreless_battery_sub_2000_500') + os.sep +  '%s.jpg'
# imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')

YEAR = '2007'

devkit_path = args.save_folder
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff



def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        # print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue

                # the VOCdevkit expects a1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))



def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    # //
    # //
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w, og_im = dataset.pull_item(i)
        # 这里im的颜色偏暗，因为BaseTransform减去了一个mean
        # im_saver = cv2.resize(im[(a2,a1,0),:,:].permute((a1,a2,0)).numpy(), (w,h))

        im_det = og_im.copy()
        im_gt = og_im.copy()

        # print(im_det)
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        # //
        # //
        # print(detections)
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            # print(boxes)
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

            # print(all_boxes)
            for item in cls_dets:
                # print(item)
                # print(item[5])
                if item[4] > thresh:
                    # print(item)
                    chinese = labelmap[j - 1] + str(round(item[4], 2))
                    # print(chinese+'det\n\n')
                    if chinese[0] == '带':
                        chinese = 'P_Battery_Core' + chinese[6:]
                    else:
                        chinese = 'P_Battery_No_Core' + chinese[7:]
                    cv2.rectangle(im_det, (item[0], item[1]), (item[2], item[3]), (0, 0, 255), 2)
                    cv2.putText(im_det, chinese, (int(item[0]), int(item[1]) - 5), 0,
                                0.6, (0, 0, 255), 2)
        real = 0
        if gt[0][4] == 3:
            real = 0
        else:
            real = 1

        for item in gt:
            if real == 0:
                print('this pic dont have the obj:', dataset.ids[i])
                break
            chinese = labelmap[int(item[4])]
            # print(chinese+'gt\n\n')
            if chinese[0] == '带':
                chinese = 'P_Battery_Core'
            else:
                chinese = 'P_Battery_No_Core'
            cv2.rectangle(im_det, (int(item[0] * w), int(item[1] * h)), (int(item[2] * w), int(item[3] * h)),
                          (0, 255, 255), 2)
            cv2.putText(im_det, chinese, (int(item[0] * w), int(item[1] * h) - 5), 0, 0.6, (0, 255, 255), 2)
            # print(labelmap[int(item[4])])

            # print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

            # cv2.imwrite('/media/trs2/wuzhangjie/SSD/eval/Xray20190723/Attention/base_battery_core_bs8_V/det_images/{0}_det.jpg'.format(dataset.ids[i]), im_det)

            # cv2.imwrite('/media/dsg3/shiyufeng/eval/Xray20190723/battery_2cV_version/20epoch_network/{0}_gt.jpg'.format(dataset.ids[i]), im_gt)
            # cv2.imwrite( '/media/dsg3/husheng/eval/{0}_det.jpg'.format(dataset.ids[i]), im_det)
            # cv2.imwrite( '/media/dsg3/husheng/eval/{0}_gt.jpg'.format(dataset.ids[i]), im_gt)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    #do_python_eval(output_dir)


def reset_args(EPOCH):
    global args
    # args.trained_model = "/media/trs2/wuzhangjie/SSD/weights/Xray20190723/2019-10-18_16-23-15Xray0723_bat_core_coreless_bs8_V_resume140/ssd300_Xray20190723_{:d}.pth".format(
    #     EPOCH)
    args.trained_model = "D:/2019_fall_semester/01_MachineLearning/ssd/weights/CORE_CORELESS.pth"
    # saver_root = '/media/trs2/wuzhangjie/SSD/eval/Xray20190723/Attention/base_battery_core_coreless_bs8_V/'
    saver_root = 'D:/2019_fall_semester/01_MachineLearning/ssd/eval'
    if not os.path.exists(saver_root):
        os.mkdir(saver_root)
    args.save_folder = saver_root + '{:d}epoch_500/'.format(EPOCH)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    else:
        shutil.rmtree(args.save_folder)
        os.mkdir(args.save_folder)

    global devkit_path
    devkit_path = args.save_folder


if __name__ == '__main__':
    # EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    # EPOCHS = [85, 90, 95, 100, 105, 110, 115, 120]
    # EPOCHS = [90, 95, 100, 105, 110, 115, 120, 125]
    # EPOCHS = [x for x in range(145, 205, 5)]
    EPOCHS = [1]
    print(EPOCHS)
    for EPOCH in EPOCHS:
        reset_args(EPOCH)

        # load net
        num_classes = len(labelmap) + 1  # +a1 for background
        net = build_ssd('test', 300, num_classes)  # initialize SSD
        net.load_state_dict(torch.load(args.trained_model))
        net.eval()
        # print('Finished loading model!')
        # load data
        dataset = SIXrayDetection(args.SIXray_root, args.imagesetfile,
                                  BaseTransform(300, dataset_mean),
                                  SIXrayAnnotationTransform())
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        # evaluation

        test_net(args.save_folder, net, args.cuda, dataset,
                 BaseTransform(net.size, dataset_mean), args.top_k, 300,
                 thresh=args.confidence_threshold)





