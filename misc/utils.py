import math
import cv2
import numpy as np
import torch


# solution proposed in https://github.com/pytorch/pytorch/issues/229#issuecomment-299424875 
def flip_tensor(tensor, dim=0):
    """
    flip the tensor on the dimension dim
    """
    inv_idx = torch.arange(tensor.shape[dim] - 1, -1, -1).to(tensor.device)
    return tensor.index_select(dim, inv_idx)


#
# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
def flip_back(output_flipped, matched_parts):
    assert len(output_flipped.shape) == 4, 'output_flipped has to be [batch_size, num_joints, height, width]'

    output_flipped = flip_tensor(output_flipped, dim=-1)

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0]].clone()
        output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
        output_flipped[:, pair[1]] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints * joints_vis, joints_vis


def get_affine_transform(center, scale, pixel_std, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 1.0 * pixel_std  # It was scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, pixel_std, output_size, rot=0):
    trans = get_affine_transform(center, scale, pixel_std, rot, output_size)

    dst_img = cv2.warpAffine(
        img, trans, (int(output_size[0]), int(output_size[1])),
        flags=cv2.INTER_LINEAR
    )

    return dst_img


#
#


#
# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
def transform_preds(coords, center, scale, pixel_std, output_size):
    coords = coords.detach().cpu().numpy()
    target_coords = np.zeros(coords.shape, dtype=np.float32)
    trans = get_affine_transform(center, scale, pixel_std, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return torch.from_numpy(target_coords)


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    # assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps.device)

    preds[:, :, 0] = idx % width  # column
    preds[:, :, 1] = torch.floor(idx / width)  # row

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(post_processing, batch_heatmaps, center, scale, pixel_std):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if post_processing:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = torch.tensor(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    ).to(batch_heatmaps.device)
                    coords[n][p] += torch.sign(diff) * .25

    preds = coords.clone()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i], pixel_std, [heatmap_width, heatmap_height])

    return preds, maxvals


def calc_dists(preds, target, normalize):
    preds = preds.type(torch.float32)
    target = target.type(torch.float32)
    dists = torch.zeros((preds.shape[1], preds.shape[0])).to(preds.device)
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                # # dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                dists[c, n] = torch.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    """
    Return percentage below threshold while ignoring values with a -1
    """
    dist_cal = torch.ne(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return torch.lt(dists[dist_cal], thr).float().sum() / num_dist_cal
    else:
        return -1


def evaluate_pck_accuracy(output, target, hm_type='gaussian', thr=0.5):
    """
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    """
    idx = list(range(output.shape[1]))
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = torch.ones((pred.shape[0], 2)) * torch.tensor([h, w],
                                                             dtype=torch.float32) / 10  # Why they divide this by 10?
        norm = norm.to(output.device)
    else:
        raise NotImplementedError
    dists = calc_dists(pred, target, norm)

    acc = torch.zeros(len(idx)).to(dists.device)
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i] = dist_acc(dists[idx[i]], thr=thr)
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    return acc, avg_acc, cnt, pred, target
#
#
