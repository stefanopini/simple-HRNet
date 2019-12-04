import os
import sys
import argparse
import ast
import csv
import cv2
import time
import torch

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import check_video_rotation


def main(filename, hrnet_c, hrnet_j, hrnet_weights, image_resolution, single_person,
         max_batch_size, csv_output_filename, csv_delimiter, device):
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

    rotation_code = check_video_rotation(filename)
    video = cv2.VideoCapture(filename)
    assert video.isOpened()
    nof_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    assert csv_output_filename.endswith('.csv')
    with open(csv_output_filename, 'wt', newline='') as fd:
        csv_output = csv.writer(fd, delimiter=csv_delimiter)

        model = SimpleHRNet(
            hrnet_c,
            hrnet_j,
            hrnet_weights,
            resolution=image_resolution,
            multiperson=not single_person,
            max_batch_size=max_batch_size,
            device=device
        )

        index = 0
        while True:
            t = time.time()

            ret, frame = video.read()
            if not ret:
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)

            pts = model.predict(frame)

            # csv format is:
            #   frame_index,detection_index,<point 0>,<point 1>,...,<point hrnet_j>
            # where each <point N> corresponds to three elements:
            #   x_coordinate,y_coordinate,confidence
            for j, pt in enumerate(pts):
                row = [index, j] + pt.flatten().tolist()
                csv_output.writerow(row)

            fps = 1. / (time.time() - t)
            print('\rframe: % 4d / %d - framerate: %f fps ' % (index, nof_frames - 1, fps), end='')

            index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='csv format is:\n'
                    '  frame_index,detection_index,<point 0>,<point 1>,...,<point hrnet_j>\n'
                    'where each <point N> corresponds to three elements:\n'
                    '  x_coordinate,y_coordinate,confidence')
    parser.add_argument("--filename", "-f", help="open the specified video",
                        type=str, default=None)
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--csv_output_filename", help="filename of the csv that will be written.", type=str,
                        default='output.csv')
    parser.add_argument("--csv_delimiter", help="csv delimiter", type=str, default=',')
    parser.add_argument("--device", help="device to be used (default: cuda, if available)", type=str, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
