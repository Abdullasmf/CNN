import torch
import torch.nn as nn
import torch.nn.functional as F

#--------------------------------------------MODEL FROM PAPER-----------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.se = SEBlock(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class ModifiedStressNet(nn.Module):
    def __init__(self):
        super(ModifiedStressNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Downsampling
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # SE-ResNet blocks
        self.res_blocks = nn.ModuleList([
            SEResNetBlock(256) for _ in range(5)
        ])
        
        # Upsampling
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Final convolution
        self.conv4 = nn.Conv2d(64, 1, kernel_size=9, padding=4)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Downsampling
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # SE-ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x)
            
        # Upsampling
        x = self.relu(self.bn4(self.deconv1(x)))
        x = self.relu(self.bn5(self.deconv2(x)))
        
        # Final convolution
        x = self.conv4(x)
        return torch.sigmoid(x)  # Scale output to 0-1

#--------------------------------------------MODEL FROM PAPER END-----------------------------------------------------
#--------------------------------------------CLAUDE-----------------------------------------------------

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DoubleConv(nn.Module):
#     """Conv => ReLU => Conv => ReLU"""
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
    
#     def forward(self, x):
#         return self.double_conv(x)

# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""
#     def __init__(self, in_channels, out_channels):
#         super(Down, self).__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )
    
#     def forward(self, x):
#         return self.maxpool_conv(x)

# class Up(nn.Module):
#     """Upscaling then double conv. Uses ConvTranspose2d for upsampling."""
#     def __init__(self, in_channels, out_channels, bilinear=False):
#         super(Up, self).__init__()
#         if bilinear:
#             # use the normal convolutions to reduce the number of channels
#             self.up = nn.Sequential(
#                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#                 nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
#             )
#         else:
#             self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
#         self.conv = DoubleConv(in_channels, out_channels)
    
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # Pad x1 if needed (in case the incomming dimensions mismatches due to pooling/upsampling)
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # Concatenate along channels
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, bilinear=False):
#         super(UNet, self).__init__()
#         self.inc = DoubleConv(in_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 1024)
#         self.up1 = Up(1024, 512, bilinear)
#         self.up2 = Up(512, 256, bilinear)
#         self.up3 = Up(256, 128, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         # Regression output layer (linear activation can be achieved by no activation)
#         self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    
#     def forward(self, x):
#         x1 = self.inc(x)      # [B, 64, 500, 500]
#         x2 = self.down1(x1)   # [B, 128, 250, 250]
#         x3 = self.down2(x2)   # [B, 256, 125, 125]
#         x4 = self.down3(x3)   # [B, 512, ~62, 62]
#         x5 = self.down4(x4)   # [B, 1024, ~31, 31]
#         x = self.up1(x5, x4)  
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         output = self.outc(x)
#         return output

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class AttentionBlock(nn.Module):
    """
    Attention Block for U-Net skip connection weighting.
    g: gating signal (from decoder)
    x: skip connection from encoder
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Up(nn.Module):
    """Upscaling then double conv with attention for the skip connection."""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # Attention block: The skip connection channels are in_channels//2 from the encoder.
        self.attention = AttentionBlock(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        # x1: decoder feature, x2: encoder feature
        x1 = self.up(x1)
        # adjust if necessary with padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x2 = self.attention(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetAttention(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=False):
        super(UNetAttention, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)      # [B, 64, 500, 500]
        x2 = self.down1(x1)   # [B, 128, 250, 250]
        x3 = self.down2(x2)   # [B, 256, 125, 125]
        x4 = self.down3(x3)   # [B, 512, ~62, 62]
        x5 = self.down4(x4)   # [B, 1024, ~31, 31]
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output

