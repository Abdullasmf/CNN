import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):  # Further reduced from 8 to 4 for finer feature capture
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reduction, bias=False),
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

class MultiscaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiscaleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels//4, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels//4, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        out = torch.cat([out1, out3, out5, out7], dim=1)
        return self.relu(self.bn(out))

class SEResNetModule(nn.Module):
    def __init__(self, channel, dilation=1):
        super(SEResNetModule, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(channel)
        self.se = SEBlock(channel)
        
        # Add edge-aware refinement
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        # Edge enhancement
        edge = self.edge_enhance(out)
        
        out = out + edge + residual
        out = self.relu(out)
        return out

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # Use more dense dilations for finer detail capture
        dilations = [1, 4, 8, 12, 16]
        
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
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilations[4], dilation=dilations[4], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )
        # Add more capacity to handle the additional ASPP branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels * 6, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x0, x1, x2, x3, x4, x5), dim=1)
        return self.conv1(x)

class EdgeDetectionModule(nn.Module):
    def __init__(self, in_channels):
        super(EdgeDetectionModule, self).__init__()
        # Sobel filters for edge detection
        self.conv_h = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        self.conv_v = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        
        # Initialize sobel filters
        with torch.no_grad():
            sobel_h = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_v = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            
            for i in range(in_channels):
                self.conv_h.weight[i, 0] = sobel_h
                self.conv_v.weight[i, 0] = sobel_v
                
        # Make weights non-trainable
        self.conv_h.weight.requires_grad = False
        self.conv_v.weight.requires_grad = False
        
        # Edge feature processing
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        edge_h = self.conv_h(x)
        edge_v = self.conv_v(x)
        edge_magnitude = torch.cat([edge_h, edge_v], dim=1)
        return self.edge_conv(edge_magnitude)

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

class StressNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(StressNet, self).__init__()
        
        # Increase feature maps for better capacity
        features = [64, 128, 256, 512, 1024]  # Added one more level for finer detail capture
        
        # Initial convolution with multiscale conv
        self.init_conv = nn.Sequential(
            MultiscaleConv(in_channels, features[0]),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        # Edge detection module at input
        self.edge_detect = EdgeDetectionModule(features[0])
        
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
        
        self.down4 = nn.Sequential(
            nn.Conv2d(features[3], features[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[4], features[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # SE-ResNet modules with different dilations for larger receptive field
        self.se_resnet = nn.Sequential(
            SEResNetModule(features[4], dilation=1),
            SEResNetModule(features[4], dilation=2),
            SEResNetModule(features[4], dilation=4),
            SEResNetModule(features[4], dilation=8),
            SEResNetModule(features[4], dilation=16),
            SEResNetModule(features[4], dilation=1)
        )
        
        # Add ASPP module for better context aggregation
        self.aspp = ASPP(features[4], features[4])
        
        # Upsampling path with new level
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[4], features[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True)
        )
        
        self.conv_after_up1 = nn.Sequential(
            nn.Conv2d(features[3] * 2, features[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            SEResNetModule(features[3])
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[3], features[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True)
        )
        
        self.conv_after_up2 = nn.Sequential(
            nn.Conv2d(features[2] * 2, features[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            SEResNetModule(features[2])
        )
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[2], features[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True)
        )
        
        self.conv_after_up3 = nn.Sequential(
            nn.Conv2d(features[1] * 2, features[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
            SEResNetModule(features[1])
        )
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features[1], features[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True)
        )
        
        self.conv_after_up4 = nn.Sequential(
            nn.Conv2d(features[0] * 2, features[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            SEResNetModule(features[0])
        )
        
        # Attention-guided refinement path for high-stress regions
        self.attention_gate1 = AttentionBlock(features[3], features[3], features[3]//2)
        self.attention_gate2 = AttentionBlock(features[2], features[2], features[2]//2)
        self.attention_gate3 = AttentionBlock(features[1], features[1], features[1]//2)
        self.attention_gate4 = AttentionBlock(features[0], features[0], features[0]//2)
        
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
        
        # Detail enhancement branch with more capacity
        self.detail_branch = nn.Sequential(
            nn.Conv2d(features[0], 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
        # Extra high-frequency detail branch using edge detection
        self.edge_detail_branch = nn.Sequential(
            EdgeDetectionModule(features[0]),
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
        edge_features = self.edge_detect(d0)
        d0 = d0 + edge_features  # Add edge features to initial features
        skip1 = d0  # First skip connection
        
        d1 = self.down1(d0)
        skip2 = d1  # Second skip connection
        
        d2 = self.down2(d1)
        skip3 = d2  # Third skip connection
        
        d3 = self.down3(d2)
        skip4 = d3  # Fourth skip connection
        
        d4 = self.down4(d3)
        
        # SE-ResNet modules
        d4 = self.se_resnet(d4)
        
        # ASPP for better context aggregation
        d4 = self.aspp(d4)
        
        # Upsampling path with attention mechanisms
        u1 = self.up1(d4)
        
        # Handle different sizes in skip connections
        if u1.shape[2:] != skip4.shape[2:]:
            u1 = F.interpolate(u1, size=skip4.shape[2:], mode='bilinear', align_corners=True)
        
        # Apply attention gating
        att_skip4 = self.attention_gate1(g=u1, x=skip4)
        u1 = torch.cat([u1, att_skip4], dim=1)
        u1 = self.conv_after_up1(u1)
        
        u2 = self.up2(u1)
        if u2.shape[2:] != skip3.shape[2:]:
            u2 = F.interpolate(u2, size=skip3.shape[2:], mode='bilinear', align_corners=True)
        
        att_skip3 = self.attention_gate2(g=u2, x=skip3)
        u2 = torch.cat([u2, att_skip3], dim=1)
        u2 = self.conv_after_up2(u2)
        
        u3 = self.up3(u2)
        if u3.shape[2:] != skip2.shape[2:]:
            u3 = F.interpolate(u3, size=skip2.shape[2:], mode='bilinear', align_corners=True)
        
        att_skip2 = self.attention_gate3(g=u3, x=skip2)
        u3 = torch.cat([u3, att_skip2], dim=1)
        u3 = self.conv_after_up3(u3)
        
        u4 = self.up4(u3)
        if u4.shape[2:] != skip1.shape[2:]:
            u4 = F.interpolate(u4, size=skip1.shape[2:], mode='bilinear', align_corners=True)
        
        att_skip1 = self.attention_gate4(g=u4, x=skip1)
        u4 = torch.cat([u4, att_skip1], dim=1)
        u4 = self.conv_after_up4(u4)
        
        # Main output
        main_out = self.final(u4)
        
        # Detail enhancement branch
        detail_out = self.detail_branch(u4)
        
        # Edge detail branch
        edge_out = self.edge_detail_branch(u4)
        
        # Combine outputs
        out = main_out + detail_out + edge_out * 0.5  # Weight edge details
        
        # Ensure output has exact same dimensions as input
        if out.shape[2:] != input_shape[2:]:
            out = F.interpolate(out, size=input_shape[2:], mode='bilinear', align_corners=True)
        
        return out

# Custom loss function for stress concentration with improved sensitivity
class StressConcentrationLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.5, threshold=0.6, edge_weight=1.5):
        super(StressConcentrationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.threshold = threshold
        self.edge_weight = edge_weight
        self.mse = nn.MSELoss(reduction='none')
        
        # Sobel filters for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def detect_edges(self, x):
        # Ensure filter is on the same device as input
        if x.is_cuda:
            self.sobel_x = self.sobel_x.cuda()
            self.sobel_y = self.sobel_y.cuda()
            
        b, c, h, w = x.size()
        x_reshaped = x.view(-1, 1, h, w)  # Reshape to apply sobel
        
        # Apply Sobel filters
        gx = F.conv2d(F.pad(x_reshaped, [1, 1, 1, 1], mode='reflect'), self.sobel_x)
        gy = F.conv2d(F.pad(x_reshaped, [1, 1, 1, 1], mode='reflect'), self.sobel_y)
        
        # Compute gradient magnitude
        edge = torch.sqrt(gx**2 + gy**2)
        return edge.view(b, c, h, w)
        
    def forward(self, pred, target):
        # Basic MSE loss
        mse_loss = self.mse(pred, target)
        
        # Create weight map based on stress magnitudes
        # Higher weight for high stress regions
        weight_map = torch.where(
            target > self.threshold, 
            1.0 + (target - self.threshold) * (1.0 / (1.0 - self.threshold)) * 1.5,  # Increased weight for high stress
            torch.ones_like(target)
        )
        
        # Detect edges in target and add additional weight
        edges = self.detect_edges(target)
        edge_weights = torch.ones_like(target) + edges * self.edge_weight
        
        # Apply focal-like weighting based on prediction error
        pt = torch.exp(-mse_loss)
        focal_weight = self.alpha * (1-pt)**self.gamma
        
        # Combine weights
        final_weight = weight_map * focal_weight * edge_weights
        
        # Apply weights to MSE loss
        weighted_loss = mse_loss * final_weight
        
        return weighted_loss.mean()