import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class LiveCameraDataset(Dataset):
    def __init__(self, camera_id=0, epoch_length=1, resolution=(384, 288), interpolation=cv2.INTER_CUBIC):
        super(LiveCameraDataset, self).__init__()
        self.camera_id = camera_id
        self.epoch_length = epoch_length
        self.resolution = resolution
        self.interpolation = interpolation

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.camera = cv2.VideoCapture(self.camera_id)
        assert self.camera.isOpened()

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, item):
        ret, frame = self.camera.read()

        if ret:
            if self.resolution is not None:
                frame = cv2.resize(frame, tuple(self.resolution), interpolation=self.interpolation)

            frame_torch = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frame_torch = None

        return ret, frame, frame_torch

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()
