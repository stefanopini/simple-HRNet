import argparse

import torch
from torch2trt import torch2trt, TRTModule

from models.hrnet import HRNet


def convert_to_trt(args):
    """
    TRT conversion function for the HRNet models using torch2trt.
    Requires the definition of the image resolution and the max batch size, supports FP16 mode (half precision).
    """
    pose = HRNet(args.hrnet_c, 17)

    pose.load_state_dict(torch.load(args.weights))
    pose.cuda().eval()

    x = torch.ones(1, 3, args.res[0], args.res[1]).cuda()
    net_trt = torch2trt(pose, [x], max_batch_size=args.batch_size, fp16_mode=args.half)
    torch.save(net_trt.state_dict(), args.output_path)
    print("Finished trt conversion!")


def parse_opt():
    """Parses the arguments for the trt conversion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", help="the model weights file", type=str,
                        default='./weights/pose_hrnet_w32_256x192.pth')
    parser.add_argument("--hrnet_c", "-c", help="HRNet channels, either 32 (default) or 48", type=int, default=32)
    parser.add_argument("--hrnet_j", "-j", help="HRNet number of joints, 17 (default)", type=int, default=17)
    parser.add_argument("--res", "-r", help="image resolution, 256x192 (default) or 384x288", type=tuple,
                        default=(256, 192))
    parser.add_argument("--batch_size", "-b", help="maximum batch size for trt", type=int, default=16)
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument("--output_path", help="output path, default ./weights/hrnet_trt.engine", type=str,
                        default="./weights/hrnet_trt.engine")
    return parser.parse_args()


def main():
    args = parse_opt()
    convert_to_trt(args)


if __name__ == '__main__':
    main()
