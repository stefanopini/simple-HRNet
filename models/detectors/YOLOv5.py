import os
import sys
import numpy as np
import torch


class YOLOv5:
    def __init__(self,
                 model_def='',
                 model_folder='./models/detectors/yolov5',
                 conf_thres=0.3,
                 device=torch.device('cpu')):

        self.model_def = model_def
        self.model_folder = model_folder
        self.conf_thres = conf_thres
        self.device = device

        # Set up model
        if self.model_def.endswith('.engine'):
            # if the yolo model ends with 'engine', it is loaded as a custom YOLOv5 pre-trained model
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', self.model_def, device=self.device)
        else:
            # load the pre-trained YOLOv5 in a pre-defined folder
            if not os.path.exists(self.model_folder):
                os.makedirs(self.model_folder)
            self.model = torch.hub.load('ultralytics/yolov5', self.model_def, pretrained=True, device=self.device)

        self.model.eval()  # Set in evaluation mode

    def predict_single(self, image, color_mode='BGR'):
        return self.predict(np.expand_dims(image.copy(), axis=0), color_mode=color_mode)[0]

    def predict(self, images, color_mode='BGR'):
        if color_mode == 'BGR':
            # all YOLO models expect RGB
            # See https://github.com/ultralytics/yolov5/issues/9913#issuecomment-1290736061
            images = images[..., ::-1]

        with torch.no_grad():
            detections = self.model(images)

            detections = detections.xyxy[0]
            detections = detections[detections[:, 4] >= self.conf_thres]

            detections = detections[detections[:, 5] == 0.]  # person

            # adding a fake class confidence to maintain compatibility with YOLOv3
            detections = torch.cat((detections[:, :5], detections[:, 4:5], detections[:, 5:]), dim=1)

            return detections
