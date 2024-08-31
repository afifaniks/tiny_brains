import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


class CustomMseLoss(nn.Module):
    def __init__(self):
        super(CustomMseLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.mean((output - target) ** 2)
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, output, target, mask):
        mask = mask.float()

        masked_output = output * mask
        masked_target = target * mask

        loss = ((masked_output - masked_target) ** 2).sum() / mask.sum()

        return loss




class CustomSsimLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super(CustomSsimLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=data_range).to(self.device)

    def forward(self, output, target):
        loss = 1 - self.ssim_loss(output, target)

        return loss


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def total_variation(self, image):
        tv_h = torch.mean(torch.abs(image[:, :, :, 1:, :] - image[:, :, :, :-1, :]))
        tv_w = torch.mean(torch.abs(image[:, :, :, :, 1:] - image[:, :, :, :, :-1]))
        return tv_h + tv_w

    def forward(self, output, target):
        tv_output = self.total_variation(output)
        tv_target = self.total_variation(target)

        loss = torch.abs(tv_output - tv_target)

        return loss
