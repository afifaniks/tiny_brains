from monai.networks import nets
from torch import nn


class Unet3DMonai(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=32):
        super(Unet3DMonai, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters

        self.model = nets.UNet(
            spatial_dims=3, in_channels=self.in_channels, out_channels=self.out_channels,
            channels=(self.num_filters, self.num_filters * 2, self.num_filters * 4, self.num_filters * 8),
            strides=(1, 1, 1)
        )

    def forward(self, inputs):
        return self.model(inputs)

