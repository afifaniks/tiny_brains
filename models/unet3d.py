import torch
import torch.nn as nn


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=32):
        super(UNet3D, self).__init__()

        # Contracting Path
        self.conv1 = self.conv_block(in_channels, num_filters)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = self.conv_block(num_filters, num_filters * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.conv3 = self.conv_block(num_filters * 2, num_filters * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        self.conv4 = self.conv_block(num_filters * 4, num_filters * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        self.conv5 = self.conv_block(num_filters * 8, num_filters * 16)

        # Expansive Path
        self.up6 = nn.ConvTranspose3d(
            num_filters * 16, num_filters * 8, kernel_size=2, stride=2
        )
        self.conv6 = self.conv_block(num_filters * 16, num_filters * 8)
        self.up7 = nn.ConvTranspose3d(
            num_filters * 8, num_filters * 4, kernel_size=2, stride=2
        )
        self.conv7 = self.conv_block(num_filters * 8, num_filters * 4)
        self.up8 = nn.ConvTranspose3d(
            num_filters * 4, num_filters * 2, kernel_size=2, stride=2
        )
        self.conv8 = self.conv_block(num_filters * 4, num_filters * 2)
        self.up9 = nn.ConvTranspose3d(
            num_filters * 2, num_filters, kernel_size=2, stride=2
        )
        self.conv9 = self.conv_block(num_filters * 2, num_filters)
        self.conv10 = nn.Conv3d(num_filters, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Contracting Path
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)

        # Expansive Path
        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(up6)
        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(up7)
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(up8)
        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(up9)
        out = self.conv10(conv9)

        return out
