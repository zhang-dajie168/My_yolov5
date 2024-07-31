import logging
import math
import os

import numpy as np
import torch
import argparse
import os
import platform
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import torch

from utils.augmentations import letterbox

from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer,
                           xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.segment.general import process_mask, masks2segments

logger = logging.getLogger()


class Yolov5_Seg(object):

    def __init__(self, save_path='', confidence_threshold=0.2, device="cuda:0", agnostic=False):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        # print(self.device)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.5
        self.agnostic = agnostic
        self.classes = None
        self.max_det = 300
        self.half = False  # use FP16 half-precision inference
        self.dnn = False
        imgsz = (640, 640)
        assert os.path.exists(save_path), 'model pth is not exits'
        # print(sys.modules)
        # device = select_device(device)
        self.net = DetectMultiBackend(save_path, device=device, dnn=self.dnn, fp16=self.half)
        stride, names, pt = self.net.stride, self.net.names, self.net.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.net.warmup(imgsz=(1 if pt or self.net.triton else 1, 3, *imgsz))

    def predict(self, img, ori_image):
        ori_size = ori_image.shape[:2]
        '''
        ori_size: h, w, _
        return: x, y, x, y, scores,  cls
        '''
        img = torch.from_numpy(img).to(self.net.device)
        img = img.half() if self.net.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        # Inference
        pred, proto = self.net(img, augment=False, visualize=False)[:2]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=self.confidence_threshold, iou_thres=self.nms_threshold,
                                   classes=self.classes, agnostic=self.agnostic,
                                   max_det=self.max_det, nm=32)
        contours = []
        det_array = []
        for i, det in enumerate(pred):
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], ori_size).round()

                segments = reversed(masks2segments(masks))
                segments = [scale_segments(img.shape[2:], x, ori_size, normalize=False) for x in segments]
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    # 获取Box列表，tensor 转numpy
                    xyxy_np = np.array([x.item() for x in xyxy])  # 使用列表推导式获取每个 Tensor 的数值，并转换为 NumPy 数组
                    conf_np = conf.item()  # 获取标量 Tensor conf 的数值
                    cls_np = cls.item()  # 获取标量 Tensor cls 的数值
                    # [x1,y1,x2,y2,conf,cls]
                    det_array.append([xyxy_np[0], xyxy_np[1], xyxy_np[2], xyxy_np[3], conf_np, cls_np])
                    # 获取轮廓列表
                    segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                    line = np.insert(segj, 0, cls_np)  # (cls,segment) size=1+n*2
                    contours.append(line)
        return contours, det_array

    def cn(self, img0):
        im = letterbox(img0, 640, stride=32, auto=True)[0]  # padded resize
        # Stack
        if len(im.shape) == 3:
            im = np.stack(im, 0)
        else:
            im = np.stack((im,) * 3, axis=-1)
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        return im


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # width
        segments[:, 1] /= img0_shape[0]  # height
    return segments


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def clip_segments(segments, shape):
    # Clip segments (xy1,xy2,...) to image shape (height, width)
    if isinstance(segments, torch.Tensor):  # faster individually
        segments[:, 0].clamp_(0, shape[1])  # x
        segments[:, 1].clamp_(0, shape[0])  # y
    else:  # np.array (faster grouped)
        segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
        segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y
