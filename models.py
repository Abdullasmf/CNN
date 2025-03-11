"""
Contains the deep learning model used for training.
This can be used on multiple scripts for training and testing.
"""
""""""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class CNNDataset(Dataset):
    # def __init__(self, image_pairs):  # image_pairs is a list of (input_image, target_image) pairs
    #     self.image_pairs = image_pairs

    # def __len__(self):
    #     return len(self.image_pairs)

    # def __getitem__(self, idx):
    #     input_image, target_image = self.image_pairs[idx]
    #     input_image = torch.tensor(input_image).unsqueeze(0).float()  # Add channel dimension
    #     target_image = torch.tensor(target_image).unsqueeze(0).float()
    #     return input_image, target_image
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_image = self.inputs[idx].float().unsqueeze(0)  # Add channel dimension
        target_image = self.targets[idx].float().unsqueeze(0)  # Add channel dimension
        return input_image, target_image






# class SEBlock(nn.Module):
#     """
#     Squeeze-and-Excitation (SE) Block
#     """
#     def __init__(self, channels, reduction_ratio=16):
#         super(SEBlock, self).__init__()
#         self.squeeze = nn.AdaptiveAvgPool2d(1)
#         self.excitation = nn.Sequential(
#             nn.Linear(channels, channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // reduction_ratio, channels),
#             nn.Sigmoid()  # Sigmoid activation for the excitation
#         )

#     def forward(self, x):
#         batch, channels, _, _ = x.size()
#         squeeze = self.squeeze(x).view(batch, channels)
#         excitation = self.excitation(squeeze).view(batch, channels, 1, 1)
#         return x * excitation  # Scale input features

# class SEResNetBottleneck(nn.Module):
#     """
#     SE-ResNet Bottleneck Block
#     """
#     def __init__(self, in_channels, out_channels, stride=1, reduction_ratio=16):
#         super(SEResNetBottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels // 4)
#         self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels // 4)
#         self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.se_block = SEBlock(out_channels, reduction_ratio)
#         self.relu = nn.ReLU(inplace=True)

#         # Shortcut connection
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#         out = self.conv3(out)
#         out = self.bn3(out)
#         out = self.se_block(out)  # Apply SE block
#         out += self.shortcut(residual)  # Add shortcut connection
#         out = self.relu(out)
#         return out

# class CNNModel(nn.Module):
#     def __init__(self):
#         super(CNNModel, self).__init__()
        
#         # Encoder (6 Conv Layers)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),  # Output: [batch, 32, 350, 500]
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # Output: [batch, 64, 175, 250]
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # Output: [batch, 128, 88, 125]
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # Output: [batch, 256, 44, 63]
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # Output: [batch, 512, 22, 32]
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
            
#             nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),  # Output: [batch, 1024, 11, 16]
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True)
#         )
        
#         # SE-ResNet Bottleneck Block
#         self.bottleneck = SEResNetBottleneck(1024, 1024)
#         self.adaptive_layer = nn.AdaptiveAvgPool2d((500, 500))
#         # Decoder (7 Deconv Layers)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, bias=False),  # Output: [batch, 512, 22, 32]
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
            
#             nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, bias=False),  # Output: [batch, 256, 44, 64]
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
            
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False),  # Output: [batch, 128, 88, 128]
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
            
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False),  # Output: [batch, 64, 176, 256]
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
            
#             nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False),  # Output: [batch, 32, 352, 512]
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
            
#             nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, bias=False),  # Output: [batch, 16, 704, 1024]
#             nn.BatchNorm2d(16),
#             nn.ReLU(inplace=True),
            
#             nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, bias=False),  # Output: [batch, 1, 1408, 2048]
#             nn.Sigmoid()  # Sigmoid activation for the final output
#         )
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.bottleneck(x)
#         x = self.decoder(x)
#         x = self.adaptive_layer(x)  # Resize output to [batch, 1, 350, 500]
#         return x
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Global average pooling
        avg_pool = torch.mean(input, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Fully connected layers to calculate scaling factor
        x = self.fc1(avg_pool)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # Excitation: scaling factor

        # Apply the scaling factor to the original input
        return input * x  # Scale the input by the excitation factor


# Define the SE-ResNet block
class SEResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SEResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.se_block = SEBlock(out_channels)
        
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.se_block(x)  # Apply SE block
        x = x + residual  # Residual connection
        return x

# Define the CNN model (including 5 SE-ResNet blocks)
class StressNet(nn.Module):
    def __init__(self):
        super(StressNet, self).__init__()

        # Downsampling
        self.c1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=4)
        self.c2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        # 5 SE-ResNet blocks
        self.se_resnet1 = SEResNetBlock(256, 256)
        self.se_resnet2 = SEResNetBlock(256, 256)
        self.se_resnet3 = SEResNetBlock(256, 256)
        self.se_resnet4 = SEResNetBlock(256, 256)
        self.se_resnet5 = SEResNetBlock(256, 256)

        # Upsampling
        self.c4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))

        # Apply 5 SE-ResNet blocks
        x = self.se_resnet1(x)
        x = self.se_resnet2(x)
        x = self.se_resnet3(x)
        x = self.se_resnet4(x)
        x = self.se_resnet5(x)

        x = torch.relu(self.c4(x))
        x = torch.relu(self.c5(x))
        x = torch.sigmoid(self.c6(x))  # Output stress field with scaled values [0,1]

        return x