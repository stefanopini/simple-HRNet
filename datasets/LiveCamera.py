import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from models.detectors.YOLOv3 import YOLOv3


class LiveCameraDataset(Dataset):
    def __init__(self, camera_id=0, epoch_length=1, resolution=(384, 288), interpolation=cv2.INTER_CUBIC,
                 multiperson=False, device=torch.device('cpu')):
        super(LiveCameraDataset, self).__init__()
        self.camera_id = camera_id
        self.epoch_length = epoch_length
        self.resolution = resolution
        self.interpolation = interpolation
        self.multiperson = multiperson
        self.device = device

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.camera = cv2.VideoCapture(self.camera_id)
        assert self.camera.isOpened()

        if self.multiperson:
            self.detector = YOLOv3(model_def="./models/detectors/yolo/config/yolov3.cfg",
                                   class_path="./models/detectors/yolo/data/coco.names",
                                   weights_path="./models/detectors/yolo/weights/yolov3.weights",
                                   classes=('person',), device=device)

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, item):
        ret, frame = self.camera.read()

        if not self.multiperson:
            if ret:
                if self.resolution is not None:
                    frame = cv2.resize(frame, tuple(self.resolution), interpolation=self.interpolation)

                frame_torch = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)

                return ret, frame, frame_torch
            else:
                return ret, []

        else:
            if ret:
                # # scale and pad image
                # ratio = min(416 / frame.shape[1], 416 / frame.shape[0])
                # imw = round(frame.shape[1] * ratio)
                # imh = round(frame.shape[0] * ratio)
                # frame_yolo = transforms.Compose([transforms.ToPILImage(),
                #                                   transforms.Resize((imh, imw)),
                #                                   transforms.Pad((max(int((imh - imw) / 2), 0),
                #                                                   max(int((imw - imh) / 2), 0),
                #                                                   max(int((imh - imw) / 2), 0),
                #                                                   max(int((imw - imh) / 2), 0)), fill=(128, 128, 128)),
                #                                   transforms.ToTensor(),
                #                             ])(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # frame_yolo = frame_yolo.unsqueeze(0).to(self.device)
                # detections = self.detector.predict(frame_yolo)
                # pad_x = max(frame.shape[1] - frame.shape[0], 0) * (416 / max(frame.shape))
                # pad_y = max(frame.shape[0] - frame.shape[1], 0) * (416 / max(frame.shape))
                # unpad_h = 416 - pad_y
                # unpad_w = 416 - pad_x

                detections = self.detector.predict_single(frame)

                if detections is not None:
                    boxes = []
                    frames = torch.zeros((len(detections), 3, self.resolution[1], self.resolution[0]))
                    for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                        # # box_h = ((y2 - y1) / unpad_h) * frame.shape[0]
                        # # box_w = ((x2 - x1) / unpad_w) * frame.shape[1]
                        # # y1 = ((y1 - pad_y // 2) / unpad_h) * frame.shape[0]
                        # # x1 = ((x1 - pad_x // 2) / unpad_w) * frame.shape[1]
                        # x1 = int(round(((x1 - pad_x // 2) / unpad_w) * frame.shape[0]))
                        # x2 = int(round(((x2 - pad_x // 2) / unpad_w) * frame.shape[0]))
                        # y1 = int(round(((y1 - pad_y // 2) / unpad_h) * frame.shape[1]))
                        # y2 = int(round(((y2 - pad_y // 2) / unpad_h) * frame.shape[1]))
                        #
                        # # x1 = int(round(x1.item() - (imw - frame.shape[1]) / 2))
                        # # x2 = int(round(x2.item() - (imw - frame.shape[1]) / 2))
                        # # y1 = int(round(y1.item() - (imh - frame.shape[0]) / 2))
                        # # y2 = int(round(y2.item() - (imh - frame.shape[0]) / 2))

                        x1 = int(round(x1.item()))
                        x2 = int(round(x2.item()))
                        y1 = int(round(y1.item()))
                        y2 = int(round(y2.item()))

                        boxes.append([x1, y1, x2, y2])
                        frames[i] = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((self.resolution[1], self.resolution[0])),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])(frame[y1:y2, x1:x2, ::-1])

                    return ret, frame, boxes, frames
                else:
                    boxes = []
                    frames = torch.zeros((0, 3, self.resolution[1], self.resolution[0]))
                    return ret, frame, boxes, frames

            else:
                return ret, [], [], []

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()
