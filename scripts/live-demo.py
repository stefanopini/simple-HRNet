import os
import argparse
import cv2
import numpy as np
import torch
from SimpleHRNet import SimpleHRNet
from misc.utils import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict


def main(camera_id, filename, hrnet_c, hrnet_j, hrnet_pretrained_weights, hrnet_joints_set, single_person,
         max_batch_size, device):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available() and True:
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    print(device)

    has_display = 'DISPLAY' in os.environ.keys()

    if filename is not None:
        video = cv2.VideoCapture(filename)
    else:
        video = cv2.VideoCapture(camera_id)
    assert video.isOpened()

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_pretrained_weights,
        multiperson=not single_person,
        max_batch_size=max_batch_size,
        device=device
    )

    while True:
        ret, frame = video.read()

        if not ret:
            break

        pts = model.predict(frame)

        for i, pt in enumerate(pts):
            frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=i,
                                             joints_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             joints_palette_samples=10)

        if has_display:
            cv2.imshow('frame.png', frame)
            cv2.waitKey(1)
        else:
            cv2.imwrite('frame.png', frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_pretrained_weights", "-w", help="hrnet parameters - path of the pretrained weights",
                        type=str, default="./pretrained_weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
