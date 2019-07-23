import os
import sys
import argparse
import ast
import cv2
import torch
from vidgear.gears import CamGear

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict


def main(camera_id, filename, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution, single_person,
         max_batch_size, disable_vidgear, device):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available() and True:
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

    print(device)

    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'

    if filename is not None:
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
    else:
        if disable_vidgear:
            video = cv2.VideoCapture(camera_id)
            assert video.isOpened()
        else:
            video = CamGear(camera_id).start()

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        resolution=image_resolution,
        multiperson=not single_person,
        max_batch_size=max_batch_size,
        device=device
    )

    while True:
        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                break
        else:
            frame = video.read()
            if frame is None:
                break

        pts = model.predict(frame)

        for i, pt in enumerate(pts):
            frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=i,
                                             points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             points_palette_samples=10)

        if has_display:
            cv2.imshow('frame.png', frame)
            k = cv2.waitKey(1)
            if k == 27:  # Esc button
                if disable_vidgear:
                    video.release()
                else:
                    video.stop()
                break
        else:
            cv2.imwrite('frame.png', frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--device", help="device to be used (default: cuda, if available)", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
