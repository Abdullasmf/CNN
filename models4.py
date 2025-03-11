import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return x * y

class SEResNetModule(nn.Module):
    def __init__(self, channel):
        super(SEResNetModule, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.se = SEBlock(channel)

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

class StressNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(StressNet, self).__init__()
        
        # Define number of features at each stage
        features = [64, 128, 256]
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling path
        self.down1 = nn.Sequential(
            nn.Conv2d(features[0], features[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(features[1], features[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # SE-ResNet modules
        self.se_resnet = nn.Sequential(
            SEResNetModule(features[2]),
            SEResNetModule(features[2]),
            SEResNetModule(features[2]),
            SEResNetModule(features[2]),
            SEResNetModule(features[2])
        )
        
        # Upsampling path
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[2], features[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True)
        )
        
        # After concatenation with skip connection (features[1] + features[1])
        self.conv_after_up1 = nn.Sequential(
            nn.Conv2d(features[1] * 2, features[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[1], features[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # After concatenation with skip connection (features[0] + features[0])
        self.conv_after_up2 = nn.Sequential(
            nn.Conv2d(features[0] * 2, features[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # Final convolutions
        self.final = nn.Sequential(
            nn.Conv2d(features[0], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        # Save input size
        input_shape = x.shape
        
        # Downsampling path with skip connections
        d0 = self.init_conv(x)
        skip1 = d0  # First skip connection
        
        d1 = self.down1(d0)
        skip2 = d1  # Second skip connection
        
        d2 = self.down2(d1)
        
        # SE-ResNet modules
        d2 = self.se_resnet(d2)
        
        # Upsampling path
        u1 = self.up1(d2)
        
        # Handle different sizes in first skip connection
        if u1.shape[2:] != skip2.shape[2:]:
            u1 = F.interpolate(u1, size=skip2.shape[2:], mode='bilinear', align_corners=True)
        
        u1 = torch.cat([u1, skip2], dim=1)
        u1 = self.conv_after_up1(u1)
        
        u2 = self.up2(u1)
        
        # Handle different sizes in second skip connection
        if u2.shape[2:] != skip1.shape[2:]:
            u2 = F.interpolate(u2, size=skip1.shape[2:], mode='bilinear', align_corners=True)
        
        u2 = torch.cat([u2, skip1], dim=1)
        u2 = self.conv_after_up2(u2)
        
        # Final convolutions
        out = self.final(u2)
        
        # Ensure output has exact same dimensions as input
        if out.shape[2:] != input_shape[2:]:
            out = F.interpolate(out, size=input_shape[2:], mode='bilinear', align_corners=True)
        
        return out