"""Contains functions for loading data"""
# pylint: disable=no-member
import logging
from functools import partial
from collections import defaultdict
from typing import Tuple, Dict, List
import torch
import numpy as np
from torch.utils.data import DataLoader

from ttt.factory import Factory


class BaseCollate:
    """Collate class for generic model to handle variable-length signals"""
    def __init__(self):
        super(BaseCollate, self).__init__()

    def __call__(self, batch: Tuple[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param batch: A tuple of dicts of processed signals and the corresponding labels
        :type batch: Tuple[Dict]
        :returns: A dict containing:
            1) tensor, batch of processed signals zero-padded and stacked on their 0 dim
            2) tensor, batch of corresponding labels
            3) list, paths to the audio files
        """
        signals = []
        labels = []
        items = []

        for data_point in batch:
            signal = data_point['signal']
            signals.append(signal)
            labels.append(data_point['label'])
            items.append(data_point['item'])

        collated_batch = {
            'signals': signals,
            'labels': torch.Tensor(labels),
            'items': items
        }

        return collated_batch

        


collate_factory = Factory()
collate_factory.register_builder('base', BaseCollate)
