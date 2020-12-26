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

from ttt.models.networks.utils import (
    _correct_state_dict, get_extractor_from_network, get_head_from_network
)
from ttt.models.networks.layers import layer_factory, ExtractorHead
from ttt.models.init import init_factory
from ttt.models.optim import optimizer_factory, scheduler_factory
from ttt.utils.typing import LayerConfigDict
# from ttt.models.networks.resnet import ResNetCIFAR as ResNet


class ClassificationModel(pl.LightningModule):
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
    def __init__(
            self, config: Dict, train_mode: str = 'train',
            val_mode: str = 'val', test_mode: str = 'test'
        ):
        super(ClassificationModel, self).__init__()
        self.train_mode = train_mode
        self.val_mode = val_mode
        self.test_mode = test_mode

        self._check_config(config)

        self.config = config
        self.network_config = config['network']

        # build and initialize the network
        self._build_network()
        self._init_network()

        # setup optimizers and schedulers
        self._setup_optimizers()

    def _build_network(self):
        """Builds the network"""
        backbone_config = self.network_config['backbone']
        backbone = layer_factory.create(
                backbone_config['name'], **backbone_config['params']
        )
        extractor = get_extractor_from_network(
            backbone, **self.network_config['extractor']['params']
        )
        main_head = get_head_from_network(
            backbone, **self.network_config['main_head']['params']
        )
        ssl_head = get_head_from_network(
            backbone, **self.network_config['ssl_head']['params']
        )

        self.main_net = ExtractorHead(extractor, main_head)
        self.ssl_net = ExtractorHead(extractor, ssl_head)

    def _init_network(self):
        """Initializes the parameters of the network"""
        pass

    def _setup_optimizers(self):
        """Sets up optimizers and schedulers"""
        kwargs = deepcopy(self.config['optimizer']['params'])
        network_params = list(self.main_net.parameters()) + list(self.ssl_net.head.parameters())
        kwargs.update({'params': network_params})
        self.optimizer = optimizer_factory.create(
            self.config['optimizer']['name'],
            **kwargs)

        if 'scheduler' in self.config['optimizer']:
            scheduler_config = deepcopy(self.config['optimizer']['scheduler'])
            scheduler_config['params']['optimizer'] = self.optimizer

            self.scheduler = scheduler_factory.create(
                scheduler_config['name'], **scheduler_config['params']
            )

    @staticmethod
    def _check_config(config):
        assert isinstance(config, dict)

        assert "network" in config
        assert set(config["network"].keys()) == {
            "backbone", "extractor", "main_head", "ssl_head"
        }

        assert "optimizer" in config


if __name__ == '__main__':
    main_classes = 10
    ssl_classes = 4
    config = {
        "network": {
            # defines the backbone network
            "backbone": {
                "name": "resnet-cifar",
                "params": {"width": 1, "depth": 26, "classes": main_classes}
            },
            # defines the extractor subnetwork from backbone
            "extractor": {"params": {"layer_name": "layer2", "add_flat": False}},
            # defines the main task head
            "main_head": {
                "params": {
                    "layer_name": "layer2",
                    "_copy": False,
                    "add_layers": [
                        {"name": "ViewFlatten", "params": {}},
                        {
                            "name": "Linear",
                            "params": {"in_features": 64 * 1, "out_features": main_classes}
                        }
                    ]
                }
            },
            # defines the SSL task head
            "ssl_head": {
                "params": {
                    "layer_name": "layer2",
                    "_copy": True,
                    "add_layers": [
                        {"name": "ViewFlatten", "params": {}},
                        {
                            "name": "Linear",
                            "params": {"in_features": 64 * 1, "out_features": ssl_classes}
                        }
                    ]
                }
            }
        },
        # define the optimizer and scheduler
        "optimizer": {
            "name": "SGD",
            "params": {
                "lr": 0.1,
                "weight_decay": 5e-4,
                "momentum": 0.9
            },
            "scheduler": {
                "name": "MultiStepLR",
                "params": {
                    "milestones": [50, 65],
                    "gamma": 0.1,
                    "last_epoch": -1
                }
            }
        }
    }
    classifier = ClassificationModel(config)

    assert classifier.main_net.extractor == classifier.ssl_net.extractor
    assert classifier.main_net.head != classifier.ssl_net.head

    assert hasattr(classifier, 'optimizer')
    assert hasattr(classifier, 'scheduler')

    import ipdb; ipdb.set_trace()
