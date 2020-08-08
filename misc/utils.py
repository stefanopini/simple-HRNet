import math
import cv2
import munkres
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
    but uses ground truth heatmap rather than y,x locations
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

    avg_acc = avg_acc / cnt if cnt != 0 else torch.tensor(0)
    return acc, avg_acc, cnt, pred, target
#
#


#
# Operations on bounding boxes (rectangles)
def bbox_area(bbox):
    """
    Area of a bounding box (a rectangles).

    Args:
        bbox (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)

    Returns:
        float: Bounding box area.
    """
    x1, y1, x2, y2 = bbox

    dx = x2 - x1
    dy = y2 - y1

    return dx * dy


def bbox_intersection(bbox_a, bbox_b):
    """
    Intersection between two buonding boxes (two rectangles).

    Args:
        bbox_a (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)
        bbox_b (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)

    Returns:
        (:class:`np.ndarray`, float):
            Intersection limits and area.

            Format: (x_min, y_min, x_max, y_max), area
    """
    x1 = np.max((bbox_a[0], bbox_b[0]))  # Left
    x2 = np.min((bbox_a[2], bbox_b[2]))  # Right
    y1 = np.max((bbox_a[1], bbox_b[1]))  # Top
    y2 = np.min((bbox_a[3], bbox_b[3]))  # Bottom

    if x2 < x1 or y2 < y1:
        bbox_i = np.asarray([0, 0, 0, 0])
        area_i = 0
    else:
        bbox_i = np.asarray([x1, y1, x2, y2], dtype=bbox_a.dtype)
        area_i = bbox_area(bbox_i)

    return bbox_i, area_i


def bbox_union(bbox_a, bbox_b):
    """
    Union between two buonding boxes (two rectangles).

    Args:
        bbox_a (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)
        bbox_b (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)

    Returns:
        float: Union.
    """
    area_a = bbox_area(bbox_a)
    area_b = bbox_area(bbox_b)

    bbox_i, area_i = bbox_intersection(bbox_a, bbox_b)
    area_u = area_a + area_b - area_i

    return area_u


def bbox_iou(bbox_a, bbox_b):
    """
    Intersection over Union (IoU) between two buonding boxes (two rectangles).

    Args:
        bbox_a (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)
        bbox_b (:class:`np.ndarray`): rectangle in the form (x_min, y_min, x_max, y_max)

    Returns:
        float: Intersection over Union (IoU).
    """
    area_u = bbox_union(bbox_a, bbox_b)
    bbox_i, area_i = bbox_intersection(bbox_a, bbox_b)

    iou = area_i / area_u

    return iou
#
#


#
# Bounding box/pose similarity and association
def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    if not isinstance(sigmas, np.ndarray):
        if d.shape[1] == 17:  # COCO
            sigmas = np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
        else:  # MPII and others
            sigmas = np.ones((d.shape[1],), dtype=np.float32) / 10.0
    vars = (sigmas * 2) ** 2
    yg = g[:, 0]
    xg = g[:, 1]
    vg = g[:, 2]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        yd = d[n_d, :, 0]
        xd = d[n_d, :, 1]
        vd = d[n_d, :, 2]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious


def compute_similarity_matrices(bboxes_a, bboxes_b, poses_a, poses_b):
    assert len(bboxes_a) == len(poses_a) and len(bboxes_b) == len(poses_b)

    result_bbox = np.zeros((len(bboxes_a), len(bboxes_b)), dtype=np.float32)
    result_pose = np.zeros((len(poses_a), len(poses_b)), dtype=np.float32)

    for i, (bbox_a, pose_a) in enumerate(zip(bboxes_a, poses_a)):
        area_bboxes_b = np.asarray([bbox_area(bbox_b) for bbox_b in bboxes_b])
        result_pose[i, :] = oks_iou(pose_a, poses_b, bbox_area(bbox_a), area_bboxes_b)
        for j, (bbox_b, pose_b) in enumerate(zip(bboxes_b, poses_b)):
            result_bbox[i, j] = bbox_iou(bbox_a, bbox_b)

    return result_bbox, result_pose


def find_person_id_associations(boxes, pts, prev_boxes, prev_pts, prev_person_ids, next_person_id=0,
                                pose_alpha=0.5, similarity_threshold=0.5, smoothing_alpha=0.):
    """
    Find associations between previous and current skeletons and apply temporal smoothing.
    It requires previous and current bounding boxes, skeletons, and previous person_ids.

    Args:
        boxes (:class:`np.ndarray`): current person bounding boxes
        pts (:class:`np.ndarray`): current human joints
        prev_boxes (:class:`np.ndarray`): previous person bounding boxes
        prev_pts (:class:`np.ndarray`): previous human joints
        prev_person_ids (:class:`np.ndarray`): previous person ids
        next_person_id (int): the id that will be assigned to the next novel detected person
            Default: 0
        pose_alpha (float): parameter to weight between bounding box similarity and pose (oks) similarity.
            pose_alpha * pose_similarity + (1 - pose_alpha) * bbox_similarity
            Default: 0.5
        similarity_threshold (float): lower similarity threshold to have a correct match between previous and
            current detections.
            Default: 0.5
        smoothing_alpha (float): linear temporal smoothing filter. Set 0 to disable, 1 to keep the previous detection.
            Default: 0.1

    Returns:
            (:class:`np.ndarray`, :class:`np.ndarray`, :class:`np.ndarray`):
                A list with (boxes, pts, person_ids) where boxes and pts are temporally smoothed.
    """
    bbox_similarity_matrix, pose_similarity_matrix = compute_similarity_matrices(boxes, prev_boxes, pts, prev_pts)
    similarity_matrix = pose_similarity_matrix * pose_alpha + bbox_similarity_matrix * (1 - pose_alpha)

    m = munkres.Munkres()
    assignments = np.asarray(m.compute((1 - similarity_matrix).tolist()))  # Munkres require a cost => 1 - similarity

    person_ids = np.ones(len(pts), dtype=np.int32) * -1
    for assignment in assignments:
        if similarity_matrix[assignment[0], assignment[1]] > similarity_threshold:
            person_ids[assignment[0]] = prev_person_ids[assignment[1]]
            if smoothing_alpha:
                boxes[assignment[0]] = (1 - smoothing_alpha) * boxes[assignment[0]] + smoothing_alpha * prev_boxes[assignment[1]]
                pts[assignment[0]] = (1 - smoothing_alpha) * pts[assignment[0]] + smoothing_alpha * prev_pts[assignment[1]]

    person_ids[person_ids == -1] = np.arange(next_person_id, next_person_id + np.sum(person_ids == -1))

    return boxes, pts, person_ids
#
#
