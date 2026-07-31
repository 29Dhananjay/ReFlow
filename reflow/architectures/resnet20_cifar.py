"""
ResNet-20 for CIFAR-10, with every block written out explicitly.

This is the definition ``chita_trained_resnet20.pt`` was pickled from -- the
attribute names (conv2_1/batchnorm2_1/downsample_conv3_1/...) are what the
checkpoint's state_dict keys refer to, so they must not be renamed or
refactored into nn.Sequential blocks.

3 stages x 3 basic blocks, 16/32/64 channels, plus 2 projection shortcuts:
21 BatchNorm2d layers total.
"""
import torch.nn as nn
import torch.nn.functional as F


class ResNet20(nn.Module):
    def __init__(self, n_classes):
        super(ResNet20, self).__init__()

        # Initial Convolution Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm1 = nn.BatchNorm2d(16)

        # Stage 1 - First block does not need downsample
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm2_1 = nn.BatchNorm2d(16)
        self.conv2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm2_2 = nn.BatchNorm2d(16)

        # Stage 1 - Subsequent blocks
        # Block 2
        self.conv2_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm2_3 = nn.BatchNorm2d(16)
        self.conv2_4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm2_4 = nn.BatchNorm2d(16)
        # Block 3
        self.conv2_5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm2_5 = nn.BatchNorm2d(16)
        self.conv2_6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm2_6 = nn.BatchNorm2d(16)

        # Stage 2 - Block 1 with downsample
        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias= False)
        self.batchnorm3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm3_2 = nn.BatchNorm2d(32)
        self.downsample_conv3_1 = nn.Conv2d(16, 32, kernel_size=1, stride=2, bias= False)
        self.downsample_bn3_1 = nn.BatchNorm2d(32)

        # Stage 2 - Subsequent blocks
        # Block 2
        self.conv3_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm3_3 = nn.BatchNorm2d(32)
        self.conv3_4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm3_4 = nn.BatchNorm2d(32)
        # Block 3
        self.conv3_5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm3_5 = nn.BatchNorm2d(32)
        self.conv3_6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm3_6 = nn.BatchNorm2d(32)

        # Stage 3 - Block 1 with downsample
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias= False)
        self.batchnorm4_1 = nn.BatchNorm2d(64)
        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm4_2 = nn.BatchNorm2d(64)
        self.downsample_conv4_1 = nn.Conv2d(32, 64, kernel_size=1, stride=2, bias= False)
        self.downsample_bn4_1 = nn.BatchNorm2d(64)

        # Stage 3 - Subsequent blocks
        # Block 2
        self.conv4_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm4_3 = nn.BatchNorm2d(64)
        self.conv4_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm4_4 = nn.BatchNorm2d(64)
        # Block 3
        self.conv4_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm4_5 = nn.BatchNorm2d(64)
        self.conv4_6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias= False)
        self.batchnorm4_6 = nn.BatchNorm2d(64)

        # Final Layers
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        # Initial convolution and batch normalization
        x = F.relu(self.batchnorm1(self.conv1(x)))
        
        # First stage (Layer 1)
        # Block 1
        identity = x
        x = F.relu(self.batchnorm2_1(self.conv2_1(x)))
        x = self.batchnorm2_2(self.conv2_2(x))
        x += identity  # This is the skip connection for the first block
        x = F.relu(x)

        # Block 2
        identity = x
        x = F.relu(self.batchnorm2_3(self.conv2_3(x)))
        x = self.batchnorm2_4(self.conv2_4(x))
        x += identity  # Skip connection for the second block
        x = F.relu(x)

        # Block 3
        identity = x
        x = F.relu(self.batchnorm2_5(self.conv2_5(x)))
        x = self.batchnorm2_6(self.conv2_6(x))
        x += identity  # Skip connection for the third block
        x = F.relu(x)

        # Second stage (Layer 2)
        # Block 1 with downsample
        identity = x
        identity = self.downsample_bn3_1(self.downsample_conv3_1(identity))
        x = F.relu(self.batchnorm3_1(self.conv3_1(x)))
        x = self.batchnorm3_2(self.conv3_2(x))
        x += identity  # Skip connection for the first block with downsampling
        x = F.relu(x)

        # Block 2
        identity = x
        x = F.relu(self.batchnorm3_3(self.conv3_3(x)))
        x = self.batchnorm3_4(self.conv3_4(x))
        x += identity  # Skip connection for the second block
        x = F.relu(x)

        # Block 3
        identity = x
        x = F.relu(self.batchnorm3_5(self.conv3_5(x)))
        x = self.batchnorm3_6(self.conv3_6(x))
        x += identity  # Skip connection for the third block
        x = F.relu(x)

        # Third stage (Layer 3)
        # Block 1 with downsample
        identity = x
        identity = self.downsample_bn4_1(self.downsample_conv4_1(identity))
        x = F.relu(self.batchnorm4_1(self.conv4_1(x)))
        x = self.batchnorm4_2(self.conv4_2(x))
        x += identity  # Skip connection for the first block with downsampling
        x = F.relu(x)

        # Block 2
        identity = x
        x = F.relu(self.batchnorm4_3(self.conv4_3(x)))
        x = self.batchnorm4_4(self.conv4_4(x))
        x += identity  # Skip connection for the second block
        x = F.relu(x)

        # Block 3
        identity = x
        x = F.relu(self.batchnorm4_5(self.conv4_5(x)))
        x = self.batchnorm4_6(self.conv4_6(x))
        x += identity  # Skip connection for the third block
        x = F.relu(x)

        # Adaptive average pooling and final fully connected layer
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the features into a vector
        x = self.fc(x)  # Fully connected layer to get class scores

        return x


