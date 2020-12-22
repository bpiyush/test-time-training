"""Defines Factory object to register various datasets"""
from ttt.factory import Factory
from ttt.data.cifar10 import CIDAR10DatasetBuilder

dataset_factory = Factory()
dataset_factory.register_builder(
    "cifar_dataset", CIDAR10DatasetBuilder())
