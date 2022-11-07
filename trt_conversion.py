from torch2trt import torch2trt,TRTModule
from models_.hrnet import HRNet
import torch
import argparse
def arg_pars():
    """
    Parses the arguments for the trt conversion.
    Args:
        weights: the weights pth file
        hrnet_c: HRNet channels either 32 or 48, default 32
        res: image resolution
        batch_size: maximum batch size for trt
        output_path: output path, default ./weights/hrnet_trt.pth
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", "-w", help="open the camera with the specified id", type=str, default='./weights/pose_hrnet_w32_256x192.pth')
    parser.add_argument("--hrnet_c", "-c", help="HRNet type, should be 32 or 48'", type=int, default=32)
    parser.add_argument("--res", "-r", type=tuple, default=(256,192))
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--output_path",type=str,default="./weights/hrnet_trt.pth")
    return parser.parse_args()
def convert_to_trt(args):
    """
    TRT conversion function using torch2trt.
    Only for the hrnet models.
    Max batch size matches simplehrnet classs max batch size.
    """
    pose = HRNet(args.hrnet_c,17)

    pose.load_state_dict(torch.load(args.weights))
    pose.cuda().eval()

    x = torch.ones(1,3,args.res[0],args.res[1]).cuda()
    net_trt = torch2trt(pose, [x],max_batch_size=args.batch_size, fp16_mode=True)
    torch.save(net_trt.state_dict(), args.output_path)
    print("Finished trt conversion!")
    
args = arg_pars()
convert_to_trt(args)
