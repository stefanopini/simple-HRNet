import os
import sys
import argparse
import ast
import cv2
import time
import torch
from vidgear.gears import CamGear
import numpy as np

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations


def main(camera_id, filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution,
         single_person, yolo_version, use_tiny_yolo, disable_tracking, max_batch_size, disable_vidgear, save_video,
         video_format, video_framerate, device, enable_tensorrt):
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
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    if filename is not None:
        rotation_code = check_video_rotation(filename)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
    else:
        rotation_code = None
        if disable_vidgear:
            video = cv2.VideoCapture(camera_id)
            assert video.isOpened()
        else:
            video = CamGear(camera_id).start()

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
        return_bounding_boxes=not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_version=yolo_version,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device,
        enable_tensorrt=enable_tensorrt
    )

    if not disable_tracking:
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0
    t_start = time.time()
    while True:
        t = time.time()

        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                t_end = time.time()
                print("\n Total Time: ", t_end - t_start)
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
        else:
            frame = video.read()
            if frame is None:
                break

        pts = model.predict(frame)

        if not disable_tracking:
            boxes, pts = pts

        if not disable_tracking:
            if len(pts) > 0:
                if prev_pts is None and prev_person_ids is None:
                    person_ids = np.arange(next_person_id, len(pts) + next_person_id, dtype=np.int32)
                    next_person_id = len(pts) + 1
                else:
                    boxes, pts, person_ids = find_person_id_associations(
                        boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts, prev_person_ids=prev_person_ids,
                        next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4, smoothing_alpha=0.1,
                    )
                    next_person_id = max(next_person_id, np.max(person_ids) + 1)
            else:
                person_ids = np.array((), dtype=np.int32)

            prev_boxes = boxes.copy()
            prev_pts = pts.copy()
            prev_person_ids = person_ids

        else:
            person_ids = np.arange(len(pts), dtype=np.int32)

        for i, (pt, pid) in enumerate(zip(pts, person_ids)):
            frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=pid,
                                             points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             points_palette_samples=10)

        # for box in boxes:
        #     cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(255,255,255),2)

        fps = 1. / (time.time() - t)
        print('\rframerate: %f fps, for %d person(s) ' % (fps,len(pts)), end='')

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

        if save_video:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
                video_writer = cv2.VideoWriter('output.avi', fourcc, video_framerate, (frame.shape[1], frame.shape[0]))
            video_writer.write(frame)

    if save_video:
        video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=48)
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
    parser.add_argument("--yolo_version",
                        help="Use the specified version of YOLO. Supported versions: `v3` (default), `v5`.",
                        type=str, default="v3")
    parser.add_argument("--use_tiny_yolo",
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection) if `yolo_version` is `v3`."
                             "Use YOLOv5n(ano) in place of YOLOv5m(edium) if `yolo_version` is `v5`."
                             "Ignored if --single_person",
                        action="store_true")
    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--save_video", help="save output frames into a video.", action="store_true")
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                               "See http://www.fourcc.org/codecs.php", type=str, default='MJPG')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
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
