import os
import sys
import argparse
import ast
import csv
import cv2
import json
import time
import torch

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import check_video_rotation


def main(format, filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, image_resolution, single_person, yolo_version,
         use_tiny_yolo, max_batch_size, csv_output_filename, csv_delimiter, json_output_filename, device,
         enable_tensorrt):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # print(device)

    image_resolution = ast.literal_eval(image_resolution)

    rotation_code = check_video_rotation(filename)
    video = cv2.VideoCapture(filename)
    assert video.isOpened()
    nof_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    assert format in ('csv', 'json')
    if format == 'csv':
        assert csv_output_filename.endswith('.csv')
        fd = open(csv_output_filename, 'wt', newline='')
        csv_output = csv.writer(fd, delimiter=csv_delimiter)
    elif format == 'json':
        assert json_output_filename.endswith('.json')
        fd = open(json_output_filename, 'wt')
        json_data = {}

    if yolo_version == 'v3':
        if use_tiny_yolo:
            yolo_model_def = "./models_/detectors/yolo/config/yolov3-tiny.cfg"
            yolo_weights_path = "./models_/detectors/yolo/weights/yolov3-tiny.weights"
        else:
            yolo_model_def = "./models_/detectors/yolo/config/yolov3.cfg"
            yolo_weights_path = "./models_/detectors/yolo/weights/yolov3.weights"
        yolo_class_path = "./models_/detectors/yolo/data/coco.names"
    elif yolo_version == 'v5':
        # YOLOv5 comes in different sizes: n(ano), s(mall), m(edium), l(arge), x(large)
        if use_tiny_yolo:
            yolo_model_def = "yolov5n"  # this  is the nano version
        else:
            yolo_model_def = "yolov5m"  # this  is the medium version
        if enable_tensorrt:
            yolo_trt_filename = yolo_model_def + ".engine"
            if os.path.exists(yolo_trt_filename):
                yolo_model_def = yolo_trt_filename
        yolo_class_path = ""
        yolo_weights_path = ""
    else:
        raise ValueError('Unsopported YOLO version.')

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        max_batch_size=max_batch_size,
        yolo_version=yolo_version,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device,
        enable_tensorrt=enable_tensorrt
    )

    index = 0
    t_start = time.time()
    while True:
        t = time.time()

        ret, frame = video.read()
        if not ret:
            t_end = time.time()
            print("\n Total Time: ", t_end - t_start)
            break
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)

        pts = model.predict(frame)

        # csv format is:
        #   frame_index,detection_index,<point 0>,<point 1>,...,<point hrnet_j>
        # where each <point N> corresponds to three elements:
        #   y_coordinate,x_coordinate,confidence

        # json format is:
        #   {frame_index: [[<point 0>,<point 1>,...,<point hrnet_j>], ...], ...}
        # where each <point N> corresponds to three elements:
        #   [y_coordinate,x_coordinate,confidence]

        if format == 'csv':
            for j, pt in enumerate(pts):
                row = [index, j] + pt.flatten().tolist()
                csv_output.writerow(row)
        elif format == 'json':
            json_data[index] = list()
            for j, pt in enumerate(pts):
                json_data[index].append(pt.tolist())

        fps = 1. / (time.time() - t)
        print('\rframe: % 4d / %d - framerate: %f fps ' % (index, nof_frames - 1, fps), end='')

        index += 1

    if format == 'json':
        json.dump(json_data, fd)

    fd.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract and save keypoints in csv or json format.\n'
                    'csv format is:\n'
                    '  frame_index,detection_index,<point 0>,<point 1>,...,<point hrnet_j>\n'
                    'where each <point N> corresponds to three elements:\n'
                    '  y_coordinate,x_coordinate,confidence\n'
                    'json format is:\n'
                    '  {frame_index: [[<point 0>,<point 1>,...,<point hrnet_j>], ...], ...}\n'
                    'where each <point N> corresponds to three elements:\n'
                    '[y_coordinate,x_coordinate,confidence]')
    parser.add_argument("--format", help="output file format. CSV or JSON.",
                        type=str, default=None)
    parser.add_argument("--filename", "-f", help="open the specified video",
                        type=str, default=None)
    parser.add_argument("--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--yolo_version",
                        help="Use the specified version of YOLO. Supported versions: `v3` (default), `v5`.",
                        type=str, default="v3")
    parser.add_argument("--use_tiny_yolo",
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection) if `yolo_version` is `v3`."
                             "Use YOLOv5n(ano) in place of YOLOv5m(edium) if `yolo_version` is `v5`."
                             "Ignored if --single_person",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--csv_output_filename", help="filename of the csv that will be written.",
                        type=str, default='output.csv')
    parser.add_argument("--csv_delimiter", help="csv delimiter", type=str, default=',')
    parser.add_argument("--json_output_filename", help="filename of the json file that will be written.",
                        type=str, default='output.json')
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    parser.add_argument("--enable_tensorrt",
                        help="Enables tensorrt inference for HRnet. If enabled, a `.engine` file is expected as "
                             "weights (`--hrnet_weights`). This option should be used only after the HRNet engine "
                             "file has been generated using the script `scripts/export-tensorrt-model.py`.",
                        action='store_true')

    args = parser.parse_args()
    main(**args.__dict__)
