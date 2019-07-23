## Multi-person Human Pose Estimation with HRNet in PyTorch

This is an unofficial implementation of the paper
 [*Deep High-Resolution Representation Learning for Human Pose Estimation*](https://arxiv.org/abs/1902.09212).  
The code is a simplified version of the [official code](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
 with the ease-of-use in mind.

The code is fully compatible with the
 [official pre-trained weights](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) and the results are the
 same of the original implementation (only slight differences on gpu due to CUDA).


This repository provides:
- A simple ``HRNet`` implementation in PyTorch (>=1.0) - compatible with official weights.
- A simple class (``SimpleHRNet``) that loads the HRNet network for the human pose estimation, loads the pre-trained weights,
 and make human predictions on a single image or a batch of images.
- Multi-person support with
 [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/47b7c912877ca69db35b8af3a38d6522681b3bb3) 
 (enabled by default).  
- A reference code that runs a live demo reading frames from a webcam or a video file.
- A relatively-simple code for training and testing the HRNet network.
- A specific script for training the network on the COCO dataset. 
 
#### Class usage

```
import cv2
from SimpleHRNet import SimpleHRNet

model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth")
image = cv2.imread("image.png", cv2.IMREAD_COLOR)

joints = model.predict(image)
```

#### Running the live demo

From a connected camera:
```
python scripts/live-demo.py --camera_id 0
```
From a saved video:
```
python scripts/live-demo.py --filename video.mp4
```

For help:
```
python scripts/live-demo.py --help
```

#### Running the training script

```
python scripts/train_coco.py
```

For help:
```
python scripts/train_coco.py --help
```

#### Requirements

- Install the required packages    
 ``pip install -r requirements.txt``
- Download the official pre-trained weights from 
[https://github.com/leoxiaobin/deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
- For multi-person support:
    - Clone
[YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/47b7c912877ca69db35b8af3a38d6522681b3bb3) 
in the folder ``./models/detectors`` and change the folder name from ``PyTorch-YOLOv3`` to ``yolo``
    - Install YOLOv3 required packages  
       ``pip install -r requirements.txt``
    - Download the pre-trained weights running the script ``download_weights.sh`` from the ``weights`` folder
    - (Optional) Download the [COCO dataset](http://cocodataset.org/#download) and save it in ``./datasets/COCO``
    - Your folders should look like:
        ```
        simple-HRNet
        ├── datasets                (datasets - for training only)
        │  └── COCO                 (COCO dataset)
        ├── losses                  (loss functions)
        ├── misc                    (misc)
        │  └── nms                  (CUDA nms module - for training only)
        ├── models                  (pytorch models)
        │  └── detectors            (people detectors)
        │    └── yolo               (PyTorch-YOLOv3 repository)
        │      ├── ...
        │      └── weights          (YOLOv3 weights)
        ├── scripts                 (scripts)
        ├── testing                 (testing code)
        ├── training                (training code)
        └── weights                 (HRnet weights)
        ```
