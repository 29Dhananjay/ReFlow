"""
MobileNetV1 with every depthwise/pointwise stage written out explicitly.

This is the architecture ``trained_MobileNetExplicit.pt`` was pickled from, so
the strides and group counts here must not be "tidied": conv16/18/20/22 are the
five repeated stride-1 512-channel stages, and conv26 is depthwise over 1024
channels (groups=1024). Changing either breaks checkpoint loading.

27 BatchNorm2d layers, which is what the variance-ratio analysis indexes over.
"""
import torch.nn as nn
import torch.nn.functional as F


class MobileNet(nn.Module):
    def __init__(self, n_class=1000):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Followed by layers according to the cfg list
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        # Following the pattern (depthwise pointwise)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn7 = nn.BatchNorm2d(128)

        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn9 = nn.BatchNorm2d(256)

        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn11 = nn.BatchNorm2d(256)

        # Stride changes on this layer according to cfg
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256, bias=False)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn13 = nn.BatchNorm2d(512)

        # Repeated 5 times according to cfg
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(512)

        # Repeating conv14 and conv15 four more times

        # Final layer before classifier according to cfg
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn16 = nn.BatchNorm2d(512)
        self.conv17 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn17 = nn.BatchNorm2d(512)

        # Final layer before classifier according to cfg
        self.conv18 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn18 = nn.BatchNorm2d(512)
        self.conv19 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn19 = nn.BatchNorm2d(512)


        # Final layer before classifier according to cfg
        self.conv20 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn20 = nn.BatchNorm2d(512)
        self.conv21 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(512)

        # Final layer before classifier according to cfg
        self.conv22 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False)
        self.bn22 = nn.BatchNorm2d(512)
        self.conv23 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn23 = nn.BatchNorm2d(512)


        # Final layer before classifier according to cfg
        self.conv24 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512, bias=False)
        self.bn24 = nn.BatchNorm2d(512)
        self.conv25 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn25 = nn.BatchNorm2d(1024)

        # Final layer before classifier according to cfg
        self.conv26 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, groups=1024, bias=False)
        self.bn26 = nn.BatchNorm2d(1024)
        self.conv27 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn27 = nn.BatchNorm2d(1024)

        
        self.classifier = nn.Linear(1024, n_class)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.relu(self.bn17(self.conv17(x)))
        x = F.relu(self.bn18(self.conv18(x)))
        x = F.relu(self.bn19(self.conv19(x)))
        x = F.relu(self.bn20(self.conv20(x)))
        x = F.relu(self.bn21(self.conv21(x)))
        x = F.relu(self.bn22(self.conv22(x)))
        x = F.relu(self.bn23(self.conv23(x)))
        x = F.relu(self.bn24(self.conv24(x)))
        x = F.relu(self.bn25(self.conv25(x)))
        x = F.relu(self.bn26(self.conv26(x)))
        x = F.relu(self.bn27(self.conv27(x)))

        x = F.avg_pool2d(x, 7)  # Global average pooling
        x = x.view(-1, 1024)  # Flatten for the classifier
        x = self.classifier(x)
        return x
