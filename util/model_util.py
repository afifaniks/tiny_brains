import torch
import torch.nn as nn

class CustomMseLoss(nn.Module):
    def __init__(self):
        super(CustomMseLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.mean((output - target) ** 2)
        return loss


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def total_variation_loss(self, image):
        tv_h = torch.mean(torch.abs(image[:, 1:, :] - image[:, :-1, :]))
        tv_w = torch.mean(torch.abs(image[:, :, 1:] - image[:, :, :-1]))
        return tv_h + tv_w

    def forward(self, output, target):
        tv_output = self.total_variation_loss(output)
        tv_target = self.total_variation_loss(target)

        loss = tv_output - tv_target

        return loss


