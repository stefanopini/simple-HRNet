## Multi-person Human Pose Estimation with HRNet in PyTorch

This is an unofficial implementation of the paper
 [*Deep High-Resolution Representation Learning for Human Pose Estimation*](https://arxiv.org/abs/1902.09212).  
The code is a simplified version of the [official code](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)
 with the ease-of-use in mind.

The code is fully compatible with the
 [official pre-trained weights](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) and the results are the
 same of the original implementation (only slight differences on gpu due to CUDA).
 It supports both Windows and Linux.


This repository provides:
- A simple ``HRNet`` implementation in PyTorch (>=1.0) - compatible with official weights (``pose_hrnet_*``).
- A simple class (``SimpleHRNet``) that loads the HRNet network for the human pose estimation, loads the pre-trained weights,
 and make human predictions on a single image or a batch of images.
- **NEW** Support for "SimpleBaselines" model based on ResNet - compatible with official weights (``pose_resnet_*``).
- **NEW** Support for multi-GPU inference.
- **NEW** Add option for using YOLOv3-tiny (faster, but less accurate person detection).
- **NEW** Add options for retrieving yolo bounding boxes and HRNet heatmaps.
- Multi-person support with
 [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/47b7c912877ca69db35b8af3a38d6522681b3bb3) 
 (enabled by default).  
- A reference code that runs a live demo reading frames from a webcam or a video file.
- A relatively-simple code for training and testing the HRNet network.
- A specific script for training the network on the COCO dataset. 
- **NEW** A [Google Colab notebook](https://github.com/stefanopini/simple-HRNet/issues/84#issuecomment-908199736) showcasing how to use this repository - Sincere thanks to [@basicvisual](https://github.com/basicvisual) and [@wuyenlin](https://github.com/wuyenlin).

If you are interested in **HigherHRNet**, please look at [*simple-HigherHRNet*](https://github.com/stefanopini/simple-HigherHRNet) 
 
### Examples

<table>
 <tr>
  <td align="center"><img src="./gifs/gif-01-output.gif" width="100%" height="auto" /></td>
  <td align="center"><img src="./gifs/gif-02-output.gif" width="100%" height="auto" /></td>
 </tr>
</table>

### Class usage

```
import cv2
from SimpleHRNet import SimpleHRNet

model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth")
image = cv2.imread("image.png", cv2.IMREAD_COLOR)

joints = model.predict(image)
```

The most useful parameters of the `__init__` function are:
<table>
 <tr>
  <td>c</td><td>number of channels (HRNet: 32, 48; PoseResNet: resnet size)</td>
 </tr>
 <tr>
  <td>nof_joints</td><td>number of joints (COCO: 17, MPII: 16)</td>
 </tr>
 <tr>
  <td>checkpoint_path</td><td>path of the (official) weights to be loaded</td>
 </tr>
 <tr>
  <td>model_name</td><td>'HRNet' or 'PoseResNet'</td>
 </tr>
 <tr>
  <td>resolution</td><td>image resolution, it depends on the loaded weights</td>
 </tr>
 <tr>
  <td>multiperson</td><td>enable multiperson prediction</td>
 </tr>
 <tr>
  <td>return_heatmaps</td><td>the `predict` method returns also the heatmaps</td>
 </tr>
 <tr>
  <td>return_bounding_boxes</td><td>the `predict` method returns also the bounding boxes (useful in conjunction with 
  `multiperson`)</td>
 </tr>
 <tr>
  <td>max_batch_size</td><td>maximum batch size used in hrnet inference</td>
 </tr>
 <tr>
  <td>device</td><td>device (cpu or cuda)</td>
 </tr>
</table>

### Running the live demo

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

### Extracting keypoints:

From a saved video:
```
python scripts/extract-keypoints.py --format csv --filename video.mp4
```

For help:
```
python scripts/extract-keypoints.py --help
```

### Running the training script

```
python scripts/train_coco.py
```

For help:
```
python scripts/train_coco.py --help
```

### Installation instructions

- Clone the repository  
 ``git clone https://github.com/stefanopini/simple-HRNet.git``
- Install the required packages  
 ``pip install -r requirements.txt``
- Download the official pre-trained weights from 
[https://github.com/leoxiaobin/deep-high-resolution-net.pytorch](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)  
  Direct links ([official Drive folder](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC), [official OneDrive folder](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ)):
  - COCO w48 384x288 (more accurate, but slower) - Used as default in `live_demo.py` and the other scripts  
    [pose_hrnet_w48_384x288.pth](https://drive.google.com/open?id=1UoJhTtjHNByZSm96W3yFTfU5upJnsKiS)
  - COCO w32 256x192 (less accurate, but faster)  
    [pose_hrnet_w32_256x192.pth](https://drive.google.com/open?id=1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38)
  - MPII w32 256x256 (MPII human joints)  
    [pose_hrnet_w32_256x256.pth](https://drive.google.com/open?id=1_wn2ifmoQprBrFvUCDedjPON4Y6jsN-v)

  Remember to set the parameters of SimpleHRNet accordingly (in particular `c`, `nof_joints`, and `resolution`).
- For multi-person support:
    - Get YOLOv3:
        - Clone [YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3/tree/47b7c912877ca69db35b8af3a38d6522681b3bb3) 
in the folder ``./models/detectors`` and change the folder name from ``PyTorch-YOLOv3`` to ``yolo``  
          OR
        - Update git submodules  
        ``git submodule update --init --recursive``
    - Install YOLOv3 required packages  
       ``pip install -r requirements.txt`` (from folder `./models/detectors/yolo`)
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
- If you want to run the training script on COCO `scripts/train_coco.py`, you have to build the `nms` module first.  
  Please note that a linux machine with CUDA is currently required. 
  Build it with either: 
  - `cd misc; make` or
  - `cd misc/nms; python setup_linux.py build_ext --inplace`  
  
  You may need to add the `./misc/nms` directory in the `PYTHONPATH` variable:  
  `export PYTHONPATH="<path-to-simple-HRNet>/misc/nms:$PYTHONPATH"`

### Google Colab notebook
Thanks to the great work of [@basicvisual](https://github.com/basicvisual) and [@wuyenlin](https://github.com/wuyenlin), you can also try this repository online on Google Colab.
More details and the notebook URL are available in [this issue](https://github.com/stefanopini/simple-HRNet/issues/84#issuecomment-908199736).  
Please make sure to make a copy on your own Google Drive and to change the Colab "Runtime type" from CPU to GPU or TPU.
