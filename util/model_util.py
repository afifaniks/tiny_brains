import torch
import torch.nn as nn
import cv2
from loguru import logger
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity,
    PeakSignalNoiseRatio
)


class CustomMseLoss(nn.Module):
    def __init__(self, scale_factor = 1):
        super(CustomMseLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.scale_factor = scale_factor

    def forward(self, output, target):
        loss = self.mse_loss(output, target) * self.scale_factor
        return loss


class MaskedMSELoss(nn.Module):
    def __init__(self, scale_factor = 1.0):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.scale_factor = scale_factor

    def forward(self, output, target, mask = None):
        if mask:
            mask = mask.float()
        else:
            mask = (target > 0.01).float()

        masked_output = output * mask
        masked_target = target * mask

        loss = self.mse_loss(masked_output, masked_target) * self.scale_factor

        return loss

class NMSELoss(nn.Module):
    def forward(self, predictions, targets):
        mse = torch.mean((predictions - targets) ** 2)
        variance = torch.var(targets)
        nmse = mse / variance
        return nmse

class CustomSsimLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super(CustomSsimLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(self.device)

    def forward(self, output, target):
        loss = 1 - self.ssim_metric(output, target)

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

class MaskedSsimLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super(MaskedSsimLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(self.device)

    def forward(self, output, target):
        coordinates = torch.nonzero(target > 0.01, as_tuple=False)

        min_coords = coordinates.min(dim=0).values
        max_coords = coordinates.max(dim=0).values

        cropped_output = output[:, :, min_coords[2]:max_coords[2], min_coords[3]:max_coords[3], min_coords[4]:max_coords[4]]
        cropped_target = target[:, :, min_coords[2]:max_coords[2], min_coords[3]:max_coords[3], min_coords[4]:max_coords[4]]

        masked_ssim_loss = 1 - self.ssim_metric(cropped_output, cropped_target)

        return masked_ssim_loss


class MaskedResidualSsimLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super(MaskedResidualSsimLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(self.device)

    def forward(self, output, target):
        mask = (target > 0.01)
        non_zero_indices = torch.nonzero(mask)
        min_coords = non_zero_indices.min(dim=0).values
        max_coords = non_zero_indices.max(dim=0).values

        cropped_output = output[min_coords[0]:max_coords[0]+1, 
                        min_coords[1]:max_coords[1]+1, 
                        min_coords[2]:max_coords[2]+1,
                        min_coords[3]:max_coords[3]+1,
                        min_coords[4]:max_coords[4]+1]
    
        cropped_target = target[min_coords[0]:max_coords[0]+1, 
                        min_coords[1]:max_coords[1]+1, 
                        min_coords[2]:max_coords[2]+1,
                        min_coords[3]:max_coords[3]+1,
                        min_coords[4]:max_coords[4]+1]
        
        residue = cropped_output - cropped_target

        masked_residual_ssim_loss = 1 - self.ssim_metric(residue, cropped_target)

        return masked_residual_ssim_loss
    
class CustomPsnrLoss(nn.Module):
    def __init__(self):
        super(CustomPsnrLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.psnr_metric = PeakSignalNoiseRatio().to(self.device)

    def forward(self, output, target):
        psnr = self.psnr_metric(output, target)
        loss = 1 - (psnr / 40.0)

        return loss
    
    
class VIFLoss3D(nn.Module):
    def __init__(self, device, sigma_n_sq=2.0, epsilon=1e-12):
        super(VIFLoss3D, self).__init__()
        # self.vif = VisualInformationFidelity(sigma_n_sq=sigma_n_sq).to(device)
        self.epsilon = epsilon

    def forward(self, preds, targets):

        loss_list = []

        for i in range(preds.shape[2]):
            if targets[:, :, i, :, :].any():
                slice_vif_loss = 1 - self.vif(preds[:, :, i, :, :], targets[:, :, i, :, :])
                loss_list.append(slice_vif_loss)

        avg_vif_loss = torch.mean(torch.stack(loss_list))

        return torch.abs(avg_vif_loss) / preds.shape[2]
    

    def vif(self, preds, targets):
        # Add epsilon to avoid division by zero
        sigma_target_sq = torch.var(targets) + self.epsilon
        sigma_v_sq = torch.var(preds) + self.epsilon
        g = torch.mean(preds * targets) / sigma_target_sq
        preds_vif_scale = torch.log10(1.0 + (g**2.0) * sigma_target_sq / (sigma_v_sq + self.epsilon))
        return preds_vif_scale.mean()
    
class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()

    def forward(self, pred, target):
        laplacian_pred = cv2.Laplacian(pred, cv2.CV_64F).var() * pred.shape[2]
        laplacian_target = cv2.Laplacian(target, cv2.CV_64F).var() * target.shape[2]

        return torch.abs(laplacian_pred - laplacian_target)
    

# def pyr_downsample(x):
#     return x[:, :, ::2, ::2]


# def pyr_upsample(x, kernel, op0, op1):
#     n_channels, _, kw, kh = kernel.shape
#     return F.conv_transpose2d(x, kernel, groups=n_channels, stride=2, padding=2, output_padding=(op0, op1))

# def gauss_kernel5(channels=1, cuda=True):
#     kernel = torch.FloatTensor([[1., 4., 6., 4., 1],
#                            [4., 16., 24., 16., 4.],
#                            [6., 24., 36., 24., 6.],
#                            [4., 16., 24., 16., 4.],
#                            [1., 4., 6., 4., 1.]])
#     kernel /= 256.
#     kernel = kernel.repeat(channels, 1, 1, 1)
#     # print(kernel)
#     if cuda:
#         kernel = kernel.cuda()
#     return Variable(kernel, requires_grad=False)


# def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
#     if size % 2 != 1:
#         raise ValueError("kernel size must be uneven")
#     grid = np.float32(np.mgrid[0:size, 0:size].T)
#     gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
#     kernel = np.sum(gaussian(grid), axis=2)
#     kernel /= np.sum(kernel)
#     # repeat same kernel across depth dimension
#     kernel = np.tile(kernel, (n_channels, 1, 1))
#     # conv weight should be (out_channels, groups/in_channels, h, w),
#     # and since we have depth-separable convolution we want the groups dimension to be 1
#     kernel = torch.FloatTensor(kernel[:, None, :, :])
#     # kernel = gauss_kernel5(n_channels)
#     if cuda:
#         kernel = kernel.cuda()
#     return Variable(kernel, requires_grad=False)


# def conv_gauss(img, kernel):
#     """ convolve img with a gaussian kernel that has been built with build_gauss_kernel """
#     _, n_channels, kd, kw, kh = kernel.shape
#     logger.info(img.shape)
#     img = F.pad(img, (kd // 2, kd // 2, kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
#     return F.conv2d(img, kernel, groups=n_channels)


# def laplacian_pyramid(img, kernel, max_levels=5):
#     current = img
#     pyr = []

#     for level in range(max_levels-1):
#         filtered = conv_gauss(current, kernel)
#         diff = current - filtered
#         pyr.append(diff)
#         current = F.avg_pool2d(filtered, 2)

#     pyr.append(current) # high -> low
#     return pyr

# def laplacian_pyramid_expand(img, kernel, max_levels=5):
#     current = img
#     pyr = []
#     for level in range(max_levels):
#         # print("level: ", level)
#         filtered = conv_gauss(current, kernel)
#         down = pyr_downsample(filtered)
#         up = pyr_upsample(down, 4*kernel, 1-filtered.size(2)%2, 1-filtered.size(3)%2)

#         diff = current - up
#         pyr.append(diff)

#         current = down
#     return pyr


# class LapLoss(nn.Module):
#     def __init__(self, max_levels=5):
#         super(LapLoss, self).__init__()
#         self.max_levels = max_levels
#         self._gauss_kernel = None

#     def forward(self, input, target):
#         if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
#             self._gauss_kernel = gauss_kernel5(input.shape[1], cuda=input.is_cuda)

#         pyr_input = laplacian_pyramid_expand(input, self._gauss_kernel, self.max_levels)
#         pyr_target = laplacian_pyramid_expand(target, self._gauss_kernel, self.max_levels)
#         weights = [1, 2, 4, 8, 16]

#         # return sum(F.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
#         return sum(weights[i] * F.l1_loss(a, b) for i, (a, b) in enumerate(zip(pyr_input, pyr_target))).mean() 


