"""Defines Factory object to register various layers"""
from typing import Any, List
import math
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import Conv2d, Linear, BatchNorm2d, \
    MaxPool2d, AdaptiveAvgPool2d, Flatten, Dropout,\
    Conv1d, BatchNorm1d, MaxPool1d, AdaptiveAvgPool1d, \
    GroupNorm, Module

from ttt.factory import Factory
from ttt.models.networks.resnet import ResNetCIFAR


class ViewFlatten(Module):
    """
    Flattening layer.
    """
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ExtractorHead(Module):
    def __init__(self, extractor, head):
        super(ExtractorHead, self).__init__()
        self.extractor = extractor
        self.head = head

    def forward(self, x):
        z = self.extractor(x)
        y = self.head(z)
        return y


layer_factory = Factory()
layer_factory.register_builder('Conv2d', Conv2d)
layer_factory.register_builder('Linear', Linear)
layer_factory.register_builder('BatchNorm2d', BatchNorm2d)
layer_factory.register_builder('MaxPool2d', MaxPool2d)
layer_factory.register_builder('AdaptiveAvgPool2d', AdaptiveAvgPool2d)
layer_factory.register_builder('Flatten', Flatten)
layer_factory.register_builder('Dropout', Dropout)
layer_factory.register_builder('Conv1d', Conv1d)
layer_factory.register_builder('BatchNorm1d', BatchNorm1d)
layer_factory.register_builder('MaxPool1d', MaxPool1d)
layer_factory.register_builder('AdaptiveAvgPool1d', AdaptiveAvgPool1d)
layer_factory.register_builder('GroupNorm', GroupNorm)
layer_factory.register_builder('ViewFlatten', ViewFlatten)
layer_factory.register_builder('resnet-cifar', ResNetCIFAR)
