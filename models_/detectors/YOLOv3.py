from __future__ import division

import os
import sys
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

sys.path.append(os.path.join(os.getcwd(), 'models', 'detectors', 'yolo'))

from .yolo.models import Darknet
from .yolo.utils.utils import load_classes, non_max_suppression


def filter_classes(detections, classes):
    mask = torch.stack([torch.stack([detections[:, -1] == cls]) for cls in classes])
    mask = torch.sum(torch.squeeze(mask, dim=1), dim=0)
    return detections[mask > 0]


# derived from https://github.com/ultralytics/yolov3/
def letterbox(img, new_shape=416, color=(127.5, 127.5, 127.5), mode='auto'):
    # Resize a rectangular image to a 32 pixel multiple rectangle
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

    if mode is 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode is 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    else:
        raise NotImplementedError

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    return img, ratio, dw, dh


# derived from https://github.com/ultralytics/yolov3/
def scale_coords(coords, from_image_shape, to_image_shape):
    # Rescale coords (xyxy) from from_image_shape to to_image_shape
    gain = max(from_image_shape) / max(to_image_shape)  # gain  = old / new
    coords[:, [0, 2]] -= (from_image_shape[1] - to_image_shape[1] * gain) / 2  # x padding
    coords[:, [1, 3]] -= (from_image_shape[0] - to_image_shape[0] * gain) / 2  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords


def prepare_data(images, color_mode='BGR', new_shape=416, color=(127.5, 127.5, 127.5), mode='square'):
    images_ok = np.zeros((images.shape[0], new_shape, new_shape, 3), dtype=images[0].dtype)
    images_tensor = torch.zeros((images.shape[0], 3, new_shape, new_shape), dtype=torch.float32)
    for i in range(len(images)):
        if color_mode == 'BGR':
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        elif color_mode == 'RGB':
            pass
        else:
            raise NotImplementedError
        images_ok[i], _, _, _ = letterbox(images[i], new_shape, color, mode)

        images_tensor[i] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])(images_ok[i])

    return images_tensor


class YOLOv3:
    def __init__(self,
                 model_def="config/yolov3.cfg",
                 class_path="data/coco.names",
                 weights_path="weights/yolov3.weights",
                 conf_thres=0.2,
                 nms_thres=0.4,
                 img_size=416,
                 classes=None,
                 max_batch_size=16,
                 device=torch.device('cpu')):

        self.model_def = model_def
        self.weights_path = weights_path
        self.class_path = class_path
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.img_size = img_size
        self.max_batch_size = max_batch_size
        self.device = device

        # Set up model
        self.model = Darknet(model_def, img_size=img_size).to(self.device)

        if weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(weights_path))

        self.model.eval()  # Set in evaluation mode

        self.classes_file = load_classes(class_path)  # Extracts class labels from file
        self.classes = classes

        self.classes_id = []
        for i, c in enumerate(self.classes_file):
            if c in self.classes:
                self.classes_id.append(i)

    def predict_single(self, image, color_mode='BGR'):
        return self.predict(np.expand_dims(image.copy(), axis=0), color_mode=color_mode)[0]

    def predict(self, images, color_mode='BGR'):
        images_rescaled = prepare_data(images.copy(), color_mode=color_mode)
        with torch.no_grad():
            images_rescaled = images_rescaled.to(self.device)

            if len(images_rescaled) <= self.max_batch_size:
                detections = self.model(images_rescaled)
            else:
                detections = torch.empty((images_rescaled.shape[0], 10647, 85)).to(self.device)
                for i in range(0, len(images_rescaled), self.max_batch_size):
                    detections[i:i + self.max_batch_size] = self.model(images_rescaled[i:i + self.max_batch_size]).detach()

            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
            for i in range(len(images)):
                if detections[i] is not None:
                    detections[i] = filter_classes(detections[i], self.classes_id)
                    detections[i] = scale_coords(detections[i], images_rescaled[i].shape[1:], images[i].shape[:2])

            return detections
