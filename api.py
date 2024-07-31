
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox
import numpy as np
import torch
import logging
logger = logging.getLogger()
import sys

from yolov5.utils.general import ( non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)

class yolov(object):

    def __init__(self, path, device='cpu', dnn=False, data='./components/xx.yaml'):
        self.device = torch.device(device)
        self.half = False
        model = DetectMultiBackend(path, device=self.device, dnn=dnn, data=data, fp16=self.half)
        stride, names, pt = model.stride, model.names, model.pt
        self.names = model.names
        logger.info("CLASSES %s, %s, %s", names, stride, pt)
        self.auto = pt
        self.model = model

    def to_yolo(self, img0):
        img = [letterbox(x, 640, stride=32, auto=self.auto)[0] for x in img0]
        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        return img

    def pre_input(self, img):
        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        
        return im

    def predict(self, img, src, convert=False, conf_thres=0.7, iou_thres=0.5, classes=None, agnostic_nms=True, max_det=300):
        if type(src).__name__ == 'list':
            ori_size = src[0].shape
        else:
            ori_size = src.shape

        detimg = img.copy()
        if convert:
            if type(detimg).__name__ != 'list':
                detimg = [detimg]
            detimg = self.to_yolo(detimg)
        detimg = self.pre_input(detimg)
        det_size = detimg.shape[2:]
        if type(detimg).__name__ == 'list':
            det_size = detimg[0].shape[2:]
        with torch.no_grad():
            pred = self.model(detimg, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        res_dets = []
        for det in pred:
            if det is not None and len(det):
                xyxy = det
                xyxy[:, :4] = scale_boxes(det_size, xyxy[:, :4], ori_size).round()
                res_dets.append(xyxy[:, :6].cpu().detach().numpy())
            else:
                res_dets.append(torch.empty((0, 6)).detach().numpy())
        # if len(res_dets) >0:
        #     return res_dets[0]

        return res_dets



