"""Defines the extension for feed-forward LightningModule."""
from typing import Dict, Tuple, Any, Union, Set
from collections import OrderedDict, defaultdict
import logging
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from ttt.models.networks.utils import _correct_state_dict
from ttt.models.networks.layers import layer_factory
from ttt.models.init import init_factory
from ttt.models.optim import optimizer_factory, scheduler_factory


class NeuralNetworkModule(pl.LightningModule):
    """Extends the LightningModule for any feed-forward model
    :param config: config for defining the module
    :type config: Dict
    :param train_mode: key for the train data split, defaults to 'train'
    :type train_mode: str, optional
    :param val_mode: key for the validation data split, defaults to 'val'
    :type val_mode: str, optional
    :param test_mode: key for the test data split, defaults to 'test'
    :type test_mode: str, optional
    """
    pass
