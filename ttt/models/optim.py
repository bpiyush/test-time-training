"""Defines Factory object to register various optimizers"""
from typing import Generator
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR, \
    OneCycleLR, MultiStepLR

from ttt.factory import Factory


optimizer_factory = Factory()
optimizer_factory.register_builder('SGD', SGD)
optimizer_factory.register_builder('Adam', Adam)
optimizer_factory.register_builder('AdamW', AdamW)

scheduler_factory = Factory()
scheduler_factory.register_builder('ReduceLROnPlateau', ReduceLROnPlateau)
scheduler_factory.register_builder('StepLR', StepLR)
scheduler_factory.register_builder('MultiStepLR', MultiStepLR)
scheduler_factory.register_builder('CyclicLR', CyclicLR)
scheduler_factory.register_builder('1cycle', OneCycleLR)

