import torch
import torch.nn.functional as F
# local modules
from PerceptualSimilarity import models
import ETlossdef as loss




class perceptual_loss():
    def __init__(self, weight=1.0, net='alex', use_gpu=True, gpu_ids=[0]):
        """
        Wrapper for PerceptualSimilarity.models.PerceptualLoss
        """
        self.model = models.PerceptualLoss(net=net, use_gpu=use_gpu, gpu_ids=gpu_ids)
        self.weight = weight

    def __call__(self, pred, target, normalize=True):
        """
        pred and target are Tensors with shape N x C x H x W (C {1, 3})
        normalize scales images from [0, 1] to [-1, 1] (default: True)
        PerceptualLoss expects N x 3 x H x W.
        """
        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
        if target.shape[1] == 1:
            target = torch.cat([target, target, target], dim=1)
        dist = self.model.forward(pred, target, normalize=normalize)
        return self.weight * dist.mean()


class temporal_consistency_loss():
    def __init__(self, weight=1.0, L0=1):
        assert L0 > 0
        self.loss = loss.temporal_consistency_loss
        self.weight = weight
        self.L0 = L0

    def __call__(self, i, image1, processed1, flow, output_images=False):
        """
        flow is from image0 to image1 (reversed when passed to
        temporal_consistency_loss function)
        """
        if i >= self.L0:
            loss = self.loss(self.image0, image1, self.processed0, processed1,
                             -flow, output_images=output_images)
            if output_images:
                loss = (self.weight * loss[0], loss[1])
            else:
                loss *= self.weight
        else:
            loss = None
        self.image0 = image1
        self.processed0 = processed1
        return loss