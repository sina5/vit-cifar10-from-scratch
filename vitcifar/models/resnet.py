"""ResNet-18 adapted to 32x32 inputs, the convolutional baseline for Project 1."""

from torch import nn
from torchvision.models import resnet18

STEM_CHANNELS = 64


class CifarResNet18Builder:
    """Builds a torchvision ResNet-18 with a CIFAR stem.

    The ImageNet stem (7x7 stride-2 conv and a max-pool) shrinks a 32x32 image to 8x8
    before the first block. The CIFAR stem keeps the full resolution instead.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def build(self) -> nn.Module:
        """Return a randomly initialized model."""
        model = resnet18(weights=None, num_classes=self.num_classes)
        model.conv1 = nn.Conv2d(3, STEM_CHANNELS, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
