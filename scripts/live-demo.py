import os
import cv2
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from datasets.LiveCamera import LiveCameraDataset
from models.hrnet import HRNet
from misc.utils import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict


def main():
    if torch.cuda.is_available() and True:
        torch.backends.cudnn.deterministic = True
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)

    has_display = 'DISPLAY' in os.environ.keys()

    model = HRNet(c=48, nof_joints=17).to(device)
    model.load_state_dict(torch.load(
        './pretrained_weights/pose_hrnet_w48_384x288.pth'
    ))
    model.eval()

    ds = LiveCameraDataset(resolution=(384, 288), multiperson=True, device=device)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    while True:
        for ret, frame, boxes, inp in dl:
            frame_cv2 = frame.detach().cpu().numpy()[0]
            frame_cv2 = np.ascontiguousarray(frame_cv2, dtype=np.uint8)
            boxes = np.asarray([[b.item() for b in box] for box in boxes], dtype=np.int32)

            if inp.shape[1] > 0:
                inp = inp.to(device)
                inp = torch.squeeze(inp, 0)

                with torch.no_grad():
                    out = model(inp)

                out = out.detach().cpu().numpy()
                for i, human in enumerate(out):
                    bbox = frame_cv2[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2]]
                    pts = np.zeros((len(human), 3), dtype=np.float32)  # x, y, confidence for each joint
                    for j, joint in enumerate(human):
                        pt = np.unravel_index(np.argmax(joint), shape=(72, 96))
                        pts[j, 0] = pt[0] * 1. / 48 * bbox.shape[0]
                        pts[j, 1] = pt[1] * 1. / 64 * bbox.shape[1]
                        pts[j, 2] = joint[pt]
                    # bbox = draw_points(bbox, pts)
                    # bbox = draw_skeleton(bbox, pts, joints_dict()['coco']['skeleton'])
                    # bbox = draw_points_and_skeleton(bbox, pts, joints_dict()['coco']['skeleton'], person_index=i)
                    bbox = draw_points_and_skeleton(bbox, pts, joints_dict()['coco']['skeleton'], person_index=i,
                                                    joints_color_palette='Set3', skeleton_color_palette='Pastel2')

            if has_display:
                cv2.imshow('frame.png', frame_cv2)
                cv2.waitKey(1)
            else:
                cv2.imwrite('frame.png', frame_cv2)

            pass


if __name__ == '__main__':
    main()
