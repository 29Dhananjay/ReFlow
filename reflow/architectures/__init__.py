"""
Model definitions that torchvision does not provide.

Both classes here exist to match a local checkpoint's pickled layout, so their
attribute names and layer shapes are fixed -- see each module's docstring.
"""

from .mobilenet_v1 import MobileNet
from .resnet20_cifar import ResNet20

__all__ = ["MobileNet", "ResNet20"]
