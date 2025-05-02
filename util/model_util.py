import torch
import torch.nn as nn
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    VisualInformationFidelity
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
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target, mask = None):
        if mask:
            mask = mask.float()
        else:
            mask = (target > 0.01).float()

        masked_output = output * mask
        masked_target = target * mask

        loss = self.mse_loss(masked_output, masked_target)

        return loss




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
    

# class VIFLoss3D(nn.Module):
#     def __init__(self, device="cpu", sigma_n_sq=2.0):
#         super(VIFLoss3D, self).__init__()
#         self.vif = VisualInformationFidelity(sigma_n_sq=sigma_n_sq).to(device)

#     def forward(self, preds, targets):
#         # Initialize the total VIF loss
#         total_vif_loss = 0.0

#         # Iterate over the depth dimension
#         for i in range(preds.shape[2]):
#             # Compute VIF loss for each slice
#             pred_slice = preds[:, :, 160, :, :]
#             target_slice = targets[:, :, 160, :, :]
#             print(i)
#             print(pred_slice.shape)

#             slice_vif_loss = 1 - self.vif(pred_slice, target_slice)
#             # total_vif_loss = total_vif_loss + slice_vif_loss

#         # Average the VIF loss over all slices
#         print("Out of the loop")
#         total_vif_loss = total_vif_loss / preds.shape[2]

#         print("Ending the function")
#         return slice_vif_loss
    

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


