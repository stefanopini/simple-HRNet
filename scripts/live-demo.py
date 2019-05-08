import os
import cv2
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from datasets.LiveCamera import LiveCameraDataset
from models.hrnet import HRNet


def main():
    if torch.cuda.is_available() and True:
        torch.backends.cudnn.deterministic = True
        device = 'cuda:0'
    else:
        device = 'cpu'

    print(device)

    has_display = 'DISPLAY' in os.environ.keys()

    model = HRNet(c=48, nof_joints=17).to(device)
    model.load_state_dict(torch.load(
        './pretrained_weights/pose_hrnet_w48_384x288.pth'
    ))

    ds = LiveCameraDataset(resolution=(384, 288))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    while True:
        for ret, frame in dl:
            out = model(frame.to(device))
            print(out.shape)

            frame_cv2 = frame.detach().cpu().numpy()[0]
            # frame_cv2 = cv2.resize(np.transpose(frame_cv2, (1, 2, 0)), (96, 72), cv2.INTER_CUBIC)
            frame_cv2 = np.ascontiguousarray(np.transpose(frame_cv2, (1, 2, 0)), dtype=np.uint8)

            out_hm = out.detach().cpu().numpy()[0]
            for o in out_hm:
                pt = np.unravel_index(np.argmax(o), shape=(72, 96))
                frame_cv2 = cv2.circle(frame_cv2, (pt[1] * 4, pt[0] * 4), 4, (0, 255, 0), -1)

            if has_display:
                cv2.imshow('frame.png', frame_cv2)
                cv2.waitKey(1)
            else:
                cv2.imwrite('frame.png', frame_cv2)

            pass


if __name__ == '__main__':
    main()
