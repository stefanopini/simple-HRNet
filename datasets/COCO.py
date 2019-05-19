import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from misc.utils import fliplr_joints, affine_transform, get_affine_transform
import pickle
import random
from tqdm import tqdm

from pycocotools.coco import COCO


class COCODataset(Dataset):
    """
    COCODataset class.
    """
    def __init__(self,
                 root_path="./datasets/COCO", data_version="train2017", is_train=True, use_gt_bboxes=True, bbox_path="",
                 image_width=288, image_height=384, color_rgb=True,
                 scale=True, scale_factor=0.35, flip_prob=0.5, rotate_prob=0.5, rotation_factor=45., half_body_prob=0.3,
                 use_different_joints_weight=False, heatmap_sigma=3
                 ):
        """
        Initializes a new COCODataset object.

        Image and annotation indexes are loaded and stored in memory.
        Annotations are preprocessed to have a simple list of annotations to iterate over.

        Bounding boxes can be loaded from the ground truth or from a pickle file (in this case, no annotations are
        provided).

        Arguments:
            root_path (str): dataset root path.
                Default: ``"./datasets/COCO"``
            data_version (str): desired version/folder of COCO. Possible options are "train2017", "val2017".
                Default: ``"train2017"``
            is_train (bool): train or eval mode. If true, train mode is used.
                Default: ``True``
            use_gt_bboxes (bool): use ground truth bounding boxes. If False, bbox_path is required.
                Default: ``True``
            bbox_path (str): bounding boxes pickle file path.
                Default: ``""``
            image_width (int): image width.
                Default: ``288``
            image_height (int): image height.
                Default: ``384``
            color_rgb (bool): rgb or bgr color mode. If True, rgb color mode is used.
                Default: ``True``
            scale (bool): scale mode.
                Default: ``True``
            scale_factor (float): scale factor.
                Default: ``0.35``
            flip_prob (float): flip probability.
                Default: ``0.5``
            rotate_prob (float): rotate probability.
                Default: ``0.5``
            rotation_factor (float): rotation factor.
                Default: ``45.``
            half_body_prob (float): half body probability.
                Default: ``0.3``
            use_different_joints_weight (bool): use different joints weights.
                If true, the following joints weights will be used:
                [1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5]
                Default: ``False``
            heatmap_sigma (float): sigma of the gaussian used to create the heatmap.
                Default: ``3``
        """

        self.root_path = root_path
        self.data_version = data_version
        self.is_train = is_train
        self.use_gt_bboxes = use_gt_bboxes
        self.bbox_path = bbox_path
        self.image_width = image_width
        self.image_height = image_height
        self.color_rgb = color_rgb
        self.scale = scale  # ToDo Check
        self.scale_factor = scale_factor
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotation_factor = rotation_factor
        self.half_body_prob = half_body_prob
        self.use_different_joints_weight = use_different_joints_weight  # ToDo Check
        self.heatmap_sigma = heatmap_sigma

        self.data_path = os.path.join(self.root_path, self.data_version)
        self.annotation_path = os.path.join(
            self.root_path, 'annotations', 'person_keypoints_%s.json' % self.data_version
        )

        self.image_size = (self.image_width, self.image_height)
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.heatmap_size = (int(self.image_width / 4), int(self.image_height / 4))
        self.heatmap_type = 'gaussian'
        self.pixel_std = 200  # I don't understand the meaning of pixel_std (=200) in the original implementation

        self.nof_joints = 17
        self.nof_joints_half_body = 8
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.upper_body_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.lower_body_ids = [11, 12, 13, 14, 15, 16]
        self.joints_weight = np.asarray(
            [1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5],
            dtype=np.float32
        ).reshape((self.nof_joints, 1))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load COCO dataset - Create COCO object then load images and annotations
        self.coco = COCO(self.annotation_path)
        self.imgIds = self.coco.getImgIds()

        # Create a list of annotations and the corresponding image (each image can contain more than one detection)

        # Load bboxes and joints
        #  if self.use_gt_bboxes -> Load GT bboxes and joints
        #  else -> Load pre-predicted bboxes by a detector (as YOLOv3) and null joints

        if not self.use_gt_bboxes:
            # bboxes must be saved as the original COCO annotations
            # i.e. the format must be:
            #  bboxes = {
            #    '<imgId>': [
            #      {
            #        'id': <annId>,  # progressive id for debugging
            #        'clean_bbox': np.array([<x>, <y>, <w>, <h>])}
            #      },
            #      ...
            #    ],
            #    ...
            #  }
            with open(self.bbox_path, 'rb') as fd:
                bboxes = pickle.load(fd)

        self.data = []
        # load annotations for each image of COCO
        for imgId in tqdm(self.imgIds):

            #
            # derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
            ann_ids = self.coco.getAnnIds(imgIds=imgId, iscrowd=False)

            img = self.coco.loadImgs(imgId)[0]

            if self.use_gt_bboxes:
                objs = self.coco.loadAnns(ann_ids)

                # sanitize bboxes
                valid_objs = []
                for obj in objs:
                    # Skip non-person objects (it should never happen)
                    if obj['category_id'] != 1:
                        continue

                    # ignore objs without keypoints annotation
                    if max(obj['keypoints']) == 0:
                        continue

                    x, y, w, h = obj['bbox']
                    x1 = np.max((0, x))
                    y1 = np.max((0, y))
                    x2 = np.min((img['width'] - 1, x1 + np.max((0, w - 1))))
                    y2 = np.min((img['height'] - 1, y1 + np.max((0, h - 1))))

                    # Use only valid bounding boxes
                    if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                        obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                        valid_objs.append(obj)

                objs = valid_objs

            else:
                objs = bboxes[imgId]

            # for each annotation of this image, add the formatted annotation to self.data
            for obj in objs:
                joints = np.zeros((self.nof_joints, 2), dtype=np.float)
                joints_visibility = np.ones((self.nof_joints, 2), dtype=np.float)

                if self.use_gt_bboxes:
                    # COCO pre-processing

                    # # Moved above
                    # # Skip non-person objects (it should never happen)
                    # if obj['category_id'] != 1:
                    #     continue
                    #
                    # # ignore objs without keypoints annotation
                    # if max(obj['keypoints']) == 0:
                    #     continue

                    for pt in range(self.nof_joints):
                        joints[pt, 0] = obj['keypoints'][pt * 3 + 0]
                        joints[pt, 1] = obj['keypoints'][pt * 3 + 1]
                        t_vis = int(np.clip(obj['keypoints'][pt * 3 + 2], 0, 1))  # ToDo check correctness
                        # COCO:
                        #   if visibility == 0 -> keypoint is not in the image.
                        #   if visibility == 1 -> keypoint is in the image BUT not visible (e.g. behind an object).
                        #   if visibility == 2 -> keypoint looks clearly (i.e. it is not hidden).
                        joints_visibility[pt, 0] = t_vis
                        joints_visibility[pt, 1] = t_vis

                center, scale = self._box2cs(obj['clean_bbox'][:4])

                self.data.append({
                    'imgId': imgId,
                    'annId': obj['id'],
                    'imgPath': os.path.join(self.root_path, self.data_version, '%012d.jpg' % imgId),
                    'center': center,
                    'scale': scale,
                    'joints': joints,
                    'joints_visibility': joints_visibility,
                })
            #
            #

        # Done check if we need prepare_data -> We should not
        print('\nCOCO dataset loaded!')

    def __len__(self,):
        return len(self.data)

    # derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
    def __getitem__(self, index):
        joints_data = self.data[index].copy()

        # Read the image from disk
        image = cv2.imread(joints_data['imgPath'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if self.color_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            raise ValueError('Fail to read %s' % image)

        joints = joints_data['joints']
        joints_vis = joints_data['joints_visibility']

        c = joints_data['center']
        s = joints_data['scale']
        score = joints_data['score'] if 'score' in joints_data else 1
        r = 0

        # Apply data augmentation
        if self.is_train:
            if self.half_body_prob and \
                    random.random() < self.half_body_prob and \
                    np.sum(joints_vis[:, 0]) > self.nof_joints_half_body:
                c_half_body, s_half_body = self.half_body_transform(joints, joints_vis)

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            sf = self.scale_factor
            rf = self.rotation_factor

            if self.scale:
                s = s * np.clip(random.random()*sf + 1, 1 - sf, 1 + sf)  # A random scale factor in [1-sf, 1+sf]

            if self.rotate_prob and random.random() < self.rotate_prob:
                r = np.clip(random.random() * rf, -rf * 2, rf * 2)  # A random rotation factor in [-2 * rf, 2 * rf]
            else:
                r = 0

            if self.flip_prob and random.random() < self.flip_prob:
                image = image[:, ::-1, :]
                joints, joints_vis = fliplr_joints(joints, joints_vis, image.shape[1], self.flip_pairs)
                c[0] = image.shape[1] - c[0] - 1

        # Apply affine transform on joints and image
        trans = get_affine_transform(c, s, self.pixel_std, r, self.image_size)
        image = cv2.warpAffine(
            image,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR
        )

        for i in range(self.nof_joints):
            if joints_vis[i, 0] > 0.:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        # Convert image to tensor and normalize
        if self.transform is not None:  # I could remove this check
            image = self.transform(image)

        target, target_weight = self.generate_target(joints, joints_vis)

        # Update metadata
        joints_data['joints'] = joints
        joints_data['joints_visibility'] = joints_vis
        joints_data['center'] = c
        joints_data['scale'] = s
        joints_data['rotation'] = r
        joints_data['score'] = score

        return image, target.astype(np.float32), target_weight.astype(np.float32), joints_data

    #
    #
    # derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2,), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.nof_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if random.random() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def generate_target(self, joints, joints_vis):
        """
        :param joints:  [nof_joints, 3]
        :param joints_vis: [nof_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        target_weight = np.ones((self.nof_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        if self.heatmap_type == 'gaussian':
            target = np.zeros((self.nof_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.heatmap_sigma * 3

            for joint_id in range(self.nof_joints):
                feat_stride = np.asarray(self.image_size) / np.asarray(self.heatmap_size)
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.heatmap_sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        else:
            raise NotImplementedError

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight
    #
    #


if __name__ == '__main__':
    coco = COCODataset(rotate_prob=0., half_body_prob=0.)
    item = coco.__getitem__(0)
    print('ok!!')
    # img = np.clip(np.transpose(item[0].numpy(), (1, 2, 0))[:, :, ::-1] * np.asarray([0.229, 0.224, 0.225]) +
    #     np.asarray([0.485, 0.456, 0.406]), 0, 1) * 255
    # cv2.imwrite('./tmp.png', img.astype(np.uint8))
    # print(item[-1])
    pass
