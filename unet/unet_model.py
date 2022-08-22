""" Full assembly of the parts to form the complete network """

from .unet_parts1d import *

class UNet1D(nn.Module):
    """
    UNet model version 1:
    In this verison the output of conv network converges to the target signal.
    """

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet1D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv1d(n_channels, 64)
        self.down1 = Down1d(64, 128)
        self.down2 = Down1d(128, 256)
        self.down3 = Down1d(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down1d(512, 1024 // factor)
        self.up1 = Up1d(1024, 512 // factor, bilinear)
        self.up2 = Up1d(512, 256 // factor, bilinear)
        self.up3 = Up1d(256, 128 // factor, bilinear)
        self.up4 = Up1d(128, 64, bilinear)
        self.outc = OutConv1d(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNet1DRes(nn.Module):
    """
    UNet model version 2:
    In this verison the output of conv network converges to the DIFFERNECE between raw signal and target signal.
    """
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet1DRes, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv1d(n_channels, 64)
        self.down1 = Down1d(64, 128)
        self.down2 = Down1d(128, 256)
        self.down3 = Down1d(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down1d(512, 1024 // factor)
        self.up1 = Up1d(1024, 512 // factor, bilinear)
        self.up2 = Up1d(512, 256 // factor, bilinear)
        self.up3 = Up1d(256, 128 // factor, bilinear)
        self.up4 = Up1d(128, 64, bilinear)
        self.outc = OutConv1d(64, n_classes)

    def forward(self, x):
        xin = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output+xin