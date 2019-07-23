import torch
import torch.nn as nn


# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
def ohkm(loss, topk):
    ohkm_loss = 0.
    for i in range(loss.shape[0]):
        sub_loss = loss[i]
        topk_val, topk_idx = torch.topk(
            sub_loss, k=topk, dim=0, sorted=False
        )
        tmp_loss = torch.gather(sub_loss, 0, topk_idx)
        ohkm_loss += torch.sum(tmp_loss) / topk
    ohkm_loss /= loss.shape[0]
    return ohkm_loss


# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        """
        MSE loss between output and GT body joints

        Args:
            use_target_weight (bool): use target weight.
                WARNING! This should be always true, otherwise the loss will sum the error for non-visible joints too.
                This has not the same meaning of joint_weights in the COCO dataset.
        """
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight=None):
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                if target_weight is None:
                    raise NameError
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


# derived from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight=True, topk=8):
        """
        MSE loss between output and GT body joints

        Args:
            use_target_weight (bool): use target weight.
                WARNING! This should be always true, otherwise the loss will sum the error for non-visible joints too.
                This has not the same meaning of joint_weights in the COCO dataset.
        """
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def forward(self, output, target, target_weight):
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(0.5 * self.criterion(heatmap_pred, heatmap_gt))

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss, self.topk)
