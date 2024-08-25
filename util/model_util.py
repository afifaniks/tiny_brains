import torch
import torch.nn as nn

class CustomMseLoss(nn.Module):
    def __init__(self):
        super(CustomMseLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.mean((output - target) ** 2)
        return loss
