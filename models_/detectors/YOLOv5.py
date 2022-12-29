import os

import cv2
import numpy as np
import torch


# from https://github.com/ultralytics/yolov5
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class YOLOv5:
    def __init__(self,
                 model_def='',
                 model_folder='./models_/detectors/yolov5',
                 image_resolution=(640, 640),
                 conf_thres=0.3,
                 device=torch.device('cpu')):

        self.model_def = model_def
        self.model_folder = model_folder
        self.image_resolution = image_resolution
        self.conf_thres = conf_thres
        self.device = device
        self.trt_model = self.model_def.endswith('.engine')

        # Set up model
        if self.trt_model:
            # if the yolo model ends with 'engine', it is loaded as a custom YOLOv5 pre-trained model
            print(f"Loading custom yolov5 model {self.model_def}")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', self.model_def)
        else:
            # load the pre-trained YOLOv5 in a pre-defined folder
            if not os.path.exists(self.model_folder):
                os.makedirs(self.model_folder)
            self.model = torch.hub.load('ultralytics/yolov5', self.model_def, pretrained=True)

        self.model = self.model.to(self.device)
        self.model.eval()  # Set in evaluation mode

    def predict_single(self, image, color_mode='BGR'):
        image = image.copy()
        if self.trt_model:
            # when running with TensorRT, the image must have fixed size
            image, (ratiow, ratioh), (dw, dh) = letterbox(image, self.image_resolution, stride=self.model.stride,
                                               auto=False, scaleFill=False)  # padded resize

        if color_mode == 'BGR':
            # all YOLO models expect RGB
            # See https://github.com/ultralytics/yolov5/issues/9913#issuecomment-1290736061 and
            # https://github.com/ultralytics/yolov5/blob/8ca182613499c323a411f559b7b5ea072122c897/models/common.py#L662
            image = image[..., ::-1]

        with torch.no_grad():
            detections = self.model(image)

            detections = detections.xyxy[0]
            detections = detections[detections[:, 4] >= self.conf_thres]

            detections = detections[detections[:, 5] == 0.]  # person

            # adding a fake class confidence to maintain compatibility with YOLOv3
            detections = torch.cat((detections[:, :5], detections[:, 4:5], detections[:, 5:]), dim=1)

        if self.trt_model:
            # account for the image resize fixing the xyxy locations
            detections[:, [0, 2]] = (detections[:, [0, 2]] - dw) / ratiow
            detections[:, [1, 3]] = (detections[:, [1, 3]] - dh) / ratioh

        return detections

    def predict(self, images, color_mode='BGR'):
        raise NotImplementedError("Not currently supported.")
