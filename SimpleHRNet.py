import cv2
import numpy as np
import torch
from torchvision.transforms import transforms

from models.hrnet import HRNet
from models.detectors.YOLOv3 import YOLOv3


class SimpleHRNet:
    """
    SimpleHRNet class.

    The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
    weights, and predict the human pose on single images.
    Multi-person support with the YOLOv3 detector is also included (and enabled by default).
    """

    def __init__(self,
                 c,
                 nof_joints,
                 checkpoint_path,
                 resolution=(384, 288),
                 interpolation=cv2.INTER_CUBIC,
                 multiperson=True,
                 max_batch_size=32,
                 yolo_model_def="./models/detectors/yolo/config/yolov3.cfg",
                 yolo_class_path="./models/detectors/yolo/data/coco.names",
                 yolo_weights_path="./models/detectors/yolo/weights/yolov3.weights",
                 device=torch.device("cpu")):
        """
        Initializes a new SimpleHRNet object.
        HRNet (and YOLOv3) are initialized on the torch.device("device") and
        its (their) pre-trained weights will be loaded from disk.

        Args:
            c (int): number of channels.
            nof_joints (int): number of joints.
            checkpoint_path (str): hrnet checkpoint path.
            resolution (tuple): hrnet input resolution - format: (height, width).
                Default: (384, 288)
            interpolation (int): opencv interpolation algorithm.
                Default: cv2.INTER_CUBIC
            multiperson (bool): if True, multiperson detection will be enabled.
                This requires the use of a people detector (like YOLOv3).
                Default: True
            max_batch_size (int): maximum batch size used in hrnet inference.
                Useless without multiperson=True.
                Default: 16
            yolo_model_def (str): path to yolo model definition file.
                Default: "./models/detectors/yolo/config/yolov3.cfg"
            yolo_class_path (str): path to yolo class definition file.
                Default: "./models/detectors/yolo/data/coco.names"
            yolo_weights_path (str): path to yolo pretrained weights file.
                Default: "./models/detectors/yolo/weights/yolov3.weights.cfg"
            device (:class:`torch.device`): the hrnet (and yolo) inference will be run on this device.
                Default: torch.device("cpu")
        """

        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.interpolation = interpolation
        self.multiperson = multiperson
        self.max_batch_size = max_batch_size
        self.yolo_model_def = yolo_model_def
        self.yolo_class_path = yolo_class_path
        self.yolo_weights_path = yolo_weights_path
        self.device = device

        self.model = HRNet(c=c, nof_joints=nof_joints).to(device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        if not self.multiperson:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            self.detector = YOLOv3(model_def=yolo_model_def,
                                   class_path=yolo_class_path,
                                   weights_path=yolo_weights_path,
                                   classes=('person',),
                                   max_batch_size=self.max_batch_size,
                                   device=device)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resolution[0], self.resolution[1])),  # (height, width)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        pass

    def predict(self, image):
        """
        Predicts the human pose on a single image.

        Args:
            image (:class:`np.ndarray`):
                the image(s) on which the human pose will be estimated.

                image is expected to be in the opencv format.
                image can be:
                    - a single image with shape=(height, width, BGR color channel)
                    - a stack of n images with shape=(n, height, width, BGR color channel)

        Returns:
            `:class:np.ndarray`:
                a numpy array containing human joints for each (detected) person.

                Format:
                    if image is a single image:
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).
                    if image is a stack of n images:
                        list of n np.ndarrays with
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).

                Each joint has 3 values: (x position, y position, joint confidence)
        """
        if len(image.shape) == 3:
            return self._predict_single(image)
        elif len(image.shape) == 4:
            return self._predict_batch(image)
        else:
            raise ValueError('Wrong image format.')

    def _predict_single(self, image):
        if not self.multiperson:
            old_res = image.shape
            if self.resolution is not None:
                image = cv2.resize(
                    image,
                    (self.resolution[1], self.resolution[0]),  # (width, height)
                    interpolation=self.interpolation
                )

            images = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0)
            boxes = np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32)  # [x1, y1, x2, y2]

        else:
            detections = self.detector.predict_single(image)

            boxes = []
            if detections is not None:
                images = torch.empty((len(detections), 3, self.resolution[0], self.resolution[1]))  # (height, width)
                for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                    x1 = int(round(x1.item()))
                    x2 = int(round(x2.item()))
                    y1 = int(round(y1.item()))
                    y2 = int(round(y2.item()))

                    boxes.append([x1, y1, x2, y2])
                    images[i] = self.transform(image[y1:y2, x1:x2, ::-1])

            else:
                images = torch.empty((0, 3, self.resolution[0], self.resolution[1]))  # (height, width)

            boxes = np.asarray(boxes, dtype=np.int32)

        if images.shape[0] > 0:
            images = images.to(self.device)

            with torch.no_grad():
                if len(images) <= self.max_batch_size:
                    out = self.model(images)

                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4)
                    ).to(self.device)
                    for i in range(0, len(images), self.max_batch_size):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])

            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            # For each human, for each joint: x, y, confidence
            for i, human in enumerate(out):
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                    # 0: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                    # 1: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                    # 2: confidences
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]

        else:
            pts = np.empty((0, 0, 3), dtype=np.float32)

        return pts

    def _predict_batch(self, images):
        if not self.multiperson:
            old_res = images[0].shape

            if self.resolution is not None:
                images_tensor = torch.empty(images.shape[0], 3, self.resolution[0], self.resolution[1])
            else:
                images_tensor = torch.empty(images.shape[0], 3, images.shape[1], images.shape[2])

            for i, image in enumerate(images):
                if self.resolution is not None:
                    image = cv2.resize(
                        image,
                        (self.resolution[1], self.resolution[0]),  # (width, height)
                        interpolation=self.interpolation
                    )

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                images_tensor[i] = self.transform(image)

            images = images_tensor
            boxes = np.repeat(
                np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32), len(images), axis=0
            )  # [x1, y1, x2, y2]

        else:
            image_detections = self.detector.predict(images)

            boxes = []
            images_tensor = []
            for d, detections in enumerate(image_detections):
                image = images[d]
                boxes_image = []
                if detections is not None:
                    images_tensor_image = torch.empty(
                        (len(detections), 3, self.resolution[0], self.resolution[1]))  # (height, width)
                    for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                        x1 = int(round(x1.item()))
                        x2 = int(round(x2.item()))
                        y1 = int(round(y1.item()))
                        y2 = int(round(y2.item()))

                        boxes_image.append([x1, y1, x2, y2])
                        images_tensor_image[i] = self.transform(image[y1:y2, x1:x2, ::-1])

                else:
                    images_tensor_image = torch.empty((0, 3, self.resolution[0], self.resolution[1]))  # (height, width)

                # stack all images and boxes in single lists
                images_tensor.extend(images_tensor_image)
                boxes.extend(boxes_image)

            # convert lists into tensors/np.ndarrays
            images = torch.tensor(np.stack(images_tensor))
            boxes = np.asarray(boxes, dtype=np.int32)

        images = images.to(self.device)

        with torch.no_grad():
            if len(images) <= self.max_batch_size:
                out = self.model(images)

            else:
                out = torch.empty(
                    (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4)
                ).to(self.device)
                for i in range(0, len(images), self.max_batch_size):
                    out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])

        out = out.detach().cpu().numpy()
        pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
        # For each human, for each joint: x, y, confidence
        for i, human in enumerate(out):
            for j, joint in enumerate(human):
                pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                # 0: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                # 1: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                # 2: confidences
                pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                pts[i, j, 2] = joint[pt]

        if self.multiperson:
            # re-add the removed batch axis (n)
            pts_batch = []
            index = 0
            for detections in image_detections:
                if detections is not None:
                    pts_batch.append(pts[index:index + len(detections)])
                    index += len(detections)
                else:
                    pts_batch.append(np.zeros((0, self.nof_joints, 3), dtype=np.float32))
            pts = pts_batch

        else:
            pts = np.expand_dims(pts, axis=1)

        return pts
