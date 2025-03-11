import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):  # Reduced from 16 to capture more detailed features
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Added max pooling for better feature emphasis
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reduction, bias=False),  # Doubled input size for avg+max
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat([avg_y, max_y], dim=1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEResNetModule(nn.Module):
    def __init__(self, channel, dilation=1):  # Added dilation parameter
        super(SEResNetModule, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=dilation, dilation=dilation)
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

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        
        self.aspp0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # Replace BatchNorm with GroupNorm for 1x1 spatial dimensions
            nn.GroupNorm(num_groups=16, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.global_avg_pool(x)
        x4 = F.interpolate(x4, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x0, x1, x2, x3, x4), dim=1)
        return self.conv1(x)

class StressNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(StressNet, self).__init__()
        
        # Define number of features at each stage - increase feature maps
        features = [64, 128, 256, 512]  # Added another level for finer detail capture
        
        # Initial convolution with smaller kernel for better detail preservation
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling path with additional level
        self.down1 = nn.Sequential(
            nn.Conv2d(features[0], features[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[1], features[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(features[1], features[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[2], features[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(features[2], features[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[3], features[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # SE-ResNet modules with different dilations for larger receptive field
        self.se_resnet = nn.Sequential(
            SEResNetModule(features[3], dilation=1),
            SEResNetModule(features[3], dilation=2),
            SEResNetModule(features[3], dilation=4),
            SEResNetModule(features[3], dilation=8),
            SEResNetModule(features[3], dilation=1)
        )
        
        # Add ASPP module for better context aggregation
        self.aspp = ASPP(features[3], features[3])
        
        # Upsampling path with new level
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[3], features[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True)
        )
        
        self.conv_after_up1 = nn.Sequential(
            nn.Conv2d(features[2] * 2, features[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            SEResNetModule(features[2])  # Added SE-ResNet module here for better feature refinement
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[2], features[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True)
        )
        
        self.conv_after_up2 = nn.Sequential(
            nn.Conv2d(features[1] * 2, features[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            SEResNetModule(features[1])  # Added SE-ResNet module here as well
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[1], features[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        self.conv_after_up3 = nn.Sequential(
            nn.Conv2d(features[0] * 2, features[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            SEResNetModule(features[0])  # Added SE-ResNet module here too
        )
        
        # Attention-guided refinement path for high-stress regions
        self.attention_gate1 = AttentionBlock(features[2], features[2], features[2]//2)
        self.attention_gate2 = AttentionBlock(features[1], features[1], features[1]//2)
        self.attention_gate3 = AttentionBlock(features[0], features[0], features[0]//2)
        
        # Final convolutions - deeper with residual connections
        self.final = nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
        # Detail enhancement branch
        self.detail_branch = nn.Sequential(
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
        skip3 = d2  # Third skip connection
        
        d3 = self.down3(d2)
        
        # SE-ResNet modules
        d3 = self.se_resnet(d3)
        
        # ASPP for better context aggregation
        d3 = self.aspp(d3)
        
        # Upsampling path with attention mechanisms
        u1 = self.up1(d3)
        
        # Handle different sizes in skip connections
        if u1.shape[2:] != skip3.shape[2:]:
            u1 = F.interpolate(u1, size=skip3.shape[2:], mode='bilinear', align_corners=True)
        
        # Apply attention gating
        att_skip3 = self.attention_gate1(g=u1, x=skip3)
        u1 = torch.cat([u1, att_skip3], dim=1)
        u1 = self.conv_after_up1(u1)
        
        u2 = self.up2(u1)
        if u2.shape[2:] != skip2.shape[2:]:
            u2 = F.interpolate(u2, size=skip2.shape[2:], mode='bilinear', align_corners=True)
        
        att_skip2 = self.attention_gate2(g=u2, x=skip2)
        u2 = torch.cat([u2, att_skip2], dim=1)
        u2 = self.conv_after_up2(u2)
        
        u3 = self.up3(u2)
        if u3.shape[2:] != skip1.shape[2:]:
            u3 = F.interpolate(u3, size=skip1.shape[2:], mode='bilinear', align_corners=True)
        
        att_skip1 = self.attention_gate3(g=u3, x=skip1)
        u3 = torch.cat([u3, att_skip1], dim=1)
        u3 = self.conv_after_up3(u3)
        
        # Main output
        main_out = self.final(u3)
        
        # Detail enhancement branch
        detail_out = self.detail_branch(u3)
        
        # Combine outputs
        out = main_out + detail_out
        
        # Ensure output has exact same dimensions as input
        if out.shape[2:] != input_shape[2:]:
            out = F.interpolate(out, size=input_shape[2:], mode='bilinear', align_corners=True)
        
        return out

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
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

# Custom loss function for stress concentration
class StressConcentrationLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, threshold=0.7):
        super(StressConcentrationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self, pred, target):
        # Basic MSE loss
        mse_loss = self.mse(pred, target)
        
        # Create weight map based on stress magnitudes
        # Higher weight for high stress regions
        weight_map = torch.where(target > self.threshold, 
                                 1.0 + (target - self.threshold) * (1.0 / (1.0 - self.threshold)),
                                 torch.ones_like(target))
        
        # Apply focal-like weighting based on prediction error
        pt = torch.exp(-mse_loss)
        focal_weight = self.alpha * (1-pt)**self.gamma
        
        # Combine weights
        final_weight = weight_map * focal_weight
        
        # Apply weights to MSE loss
        weighted_loss = mse_loss * final_weight
        
        return weighted_loss.mean()