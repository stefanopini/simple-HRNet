import os
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets.HumanPoseEstimation import HumanPoseEstimationDataset
from losses.loss import JointsMSELoss, JointsOHKMMSELoss
from misc.checkpoint import load_checkpoint
from misc.utils import flip_tensor, flip_back
from misc.visualization import save_images
from models.hrnet import HRNet


class Test(object):
    """
    Test class.

    The class provides a basic tool for testing HRNet checkpoints.

    The only method supposed to be directly called is `run()`.
    """

    def __init__(self,
                 ds_test,
                 batch_size=1,
                 num_workers=4,
                 loss='JointsMSELoss',
                 checkpoint_path=None,
                 model_c=48,
                 model_nof_joints=17,
                 model_bn_momentum=0.1,
                 flip_test_images=True,
                 device=None
                 ):
        """
        Initializes a new Test object.

        The HRNet model is initialized and the saved checkpoint is loaded.
        The DataLoader and the loss function are defined.

        Args:
            ds_test (HumanPoseEstimationDataset): test dataset.
            batch_size (int): batch size.
                Default: 1
            num_workers (int): number of workers for each DataLoader
                Default: 4
            loss (str): loss function. Valid values are 'JointsMSELoss' and 'JointsOHKMMSELoss'.
                Default: "JointsMSELoss"
            checkpoint_path (str): path to a previous checkpoint.
                Default: None
            model_c (int): hrnet parameters - number of channels.
                Default: 48
            model_nof_joints (int): hrnet parameters - number of joints.
                Default: 17
            model_bn_momentum (float): hrnet parameters - path to the pretrained weights.
                Default: 0.1
            flip_test_images (bool): flip images during validating.
                Default: True
            device (torch.device): device to be used (default: cuda, if available).
                Default: None
        """
        super(Test, self).__init__()

        self.ds_test = ds_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loss = loss
        self.checkpoint_path = checkpoint_path
        self.model_c = model_c
        self.model_nof_joints = model_nof_joints
        self.model_bn_momentum = model_bn_momentum
        self.flip_test_images = flip_test_images
        self.epoch = 0

        # torch device
        if device is not None:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cpu')

        print(self.device)

        #
        # load model
        self.model = HRNet(c=self.model_c, nof_joints=self.model_nof_joints,
                           bn_momentum=self.model_bn_momentum).to(self.device)

        #
        # define loss
        if self.loss == 'JointsMSELoss':
            self.loss_fn = JointsMSELoss().to(self.device)
        elif self.loss == 'JointsOHKMMSELoss':
            self.loss_fn = JointsOHKMMSELoss().to(self.device)
        else:
            raise NotImplementedError

        #
        # load previous checkpoint
        if self.checkpoint_path is not None:
            print('Loading checkpoint %s...' % self.checkpoint_path)
            if os.path.isdir(self.checkpoint_path):
                path = os.path.join(self.checkpoint_path, 'checkpoint_last.pth')
            else:
                path = self.checkpoint_path
            self.starting_epoch, self.model, _, self.params = load_checkpoint(path, self.model, device=self.device)
        else:
            raise ValueError('checkpoint_path is not defined')

        #
        # load test dataset
        self.dl_test = DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.len_dl_test = len(self.dl_test)

        #
        # initialize variables
        self.mean_loss_test = 0.
        self.mean_acc_test = 0.

    def _test(self):
        self.model.eval()
        with torch.no_grad():
            for step, (image, target, target_weight, joints_data) in enumerate(tqdm(self.dl_test, desc='Test')):
                image = image.to(self.device)
                target = target.to(self.device)
                target_weight = target_weight.to(self.device)

                output = self.model(image)

                if self.flip_test_images:
                    image_flipped = flip_tensor(image, dim=-1)
                    output_flipped = self.model(image_flipped)

                    output_flipped = flip_back(output_flipped, self.ds_test.flip_pairs)

                    output = (output + output_flipped) * 0.5

                loss = self.loss_fn(output, target, target_weight)

                # Evaluate accuracy
                # Get predictions on the input
                accs, avg_acc, cnt, joints_preds, joints_target = \
                    self.ds_test.evaluate_accuracy(output, target)

                self.mean_loss_test += loss.item()
                self.mean_acc_test += avg_acc.item()
                if step == 0:
                    save_images(image, target, joints_target, output, joints_preds, joints_data['joints_visibility'])

        self.mean_loss_test /= self.len_dl_test
        self.mean_acc_test /= self.len_dl_test

        print('\nTest: Loss %f - Accuracy %f' % (self.mean_loss_test, self.mean_acc_test))

    def run(self):
        """
        Runs the test.
        """

        print('\nTest started @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # start testing
        print('\nLoaded checkpoint %s @ %s\nSaved epoch %d' %
              (self.checkpoint_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.starting_epoch))

        self.mean_loss_test = 0.
        self.mean_acc_test = 0.

        #
        # Test

        self._test()

        print('\nTest ended @ %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
