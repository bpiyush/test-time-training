"""Defines the extension for feed-forward LightningModule."""
from typing import Dict, Tuple, Any, Union, Set
from collections import OrderedDict, defaultdict
import logging
from abc import abstractmethod
from copy import deepcopy

from tqdm import tqdm
from termcolor import colored
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from ttt.models.networks.utils import (
    _correct_state_dict, get_extractor_from_network, get_head_from_network
)
from ttt.models.networks.layers import layer_factory, ExtractorHead
from ttt.models.init import init_factory
from ttt.models.optim import optimizer_factory, scheduler_factory
from ttt.utils.typing import LayerConfigDict
from ttt.models.losses import loss_factory
from ttt.data.data_module import DataModule
from ttt.models.utils import rotate_batch


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
        self.data_config = self.config['data']
        self.model_config = self.config['model']
        self.network_config = self.model_config['network']

        # build and initialize the network
        self._build_network()
        self._init_network()

        # setup optimizers and schedulers
        self._setup_optimizers()

        # setup losses
        self._setup_losses()

        # setup dataloaders
        self._setup_data()

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
        kwargs = deepcopy(self.model_config['optimizer']['params'])
        network_params = list(self.main_net.parameters()) + list(self.ssl_net.head.parameters())
        kwargs.update({'params': network_params})
        self.optimizer = optimizer_factory.create(
            self.model_config['optimizer']['name'],
            **kwargs)

        if 'scheduler' in self.model_config['optimizer']:
            scheduler_config = deepcopy(self.model_config['optimizer']['scheduler'])
            scheduler_config['params']['optimizer'] = self.optimizer

            self.scheduler = scheduler_factory.create(
                scheduler_config['name'], **scheduler_config['params']
            )

    def _setup_losses(self):
        self.losses = dict()
        for index, loss_config in enumerate(self.model_config['losses']):
            loss_fn = loss_factory.create(
                loss_config['name'], **loss_config['params']
            )
            self.losses.update({loss_config['name']: loss_fn})

    def _setup_data(self):
        self.data_module = DataModule(
            self.data_config,
            self.model_config['batch_size'],
            self.model_config['num_workers']
        )

    def _process_batch(self, batch, mode, rotation):
        inputs, labels, items = batch['signals'], batch['labels'], batch['items']
        ssl_inputs, ssl_labels = rotate_batch(inputs, rotation)

        if mode == "train":
            main_predictions = self.main_net(inputs)
            ssl_predictions = self.ssl_net(ssl_inputs)
        else:
            with torch.no_grad():
                main_predictions = self.main_net(inputs)
                ssl_predictions = self.ssl_net(ssl_inputs)

        batch_data = {
            "inputs": inputs,
            "main_predictions": main_predictions,
            "main_targets": labels,
            "ssl_predictions": ssl_predictions,
            "ssl_targets": ssl_labels,
            "items": items
        }

        return batch_data

    def _update_network_params(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def _process_epoch(self, dataloader, mode, epoch):

        if mode == "train":
            self.main_net.train()
            self.ssl_net.train()
        else:
            self.main_net.eval()
            self.ssl_net.eval()

        iterator = tqdm(dataloader, dynamic_ncols=True)

        for batch_idx, batch in enumerate(iterator):
            rotation = "rand" if mode == "train" else "expand"
            batch_data = self._process_batch(batch, mode, rotation)

            # main task loss
            main_loss = 0.0
            for loss_key, loss_fn in self.losses.items():
                main_loss += loss_fn(batch_data['main_predictions'], batch_data['main_targets'])

            # SSL task loss
            ssl_loss = 0.0
            for loss_key, loss_fn in self.losses.items():
                ssl_loss += loss_fn(batch_data['ssl_predictions'], batch_data['ssl_targets'])

            loss = main_loss + ssl_loss

            iterator.set_description(
                "V: {} | Epoch: {} | {} | Loss {:.4f}".format(
                    colored(self.config['version'], "blue"),
                    colored(epoch, "blue"),
                    colored(mode.upper(), "blue"),
                    loss
                ), refresh=True
            )

            if mode == "train":
                self._update_network_params(loss)

    def fit(self, n_epochs: int, train_dataloader, test_dataloader):
        for epoch in range(n_epochs):
            self._process_epoch(train_dataloader, "train", epoch)
            self._process_epoch(test_dataloader, "test", epoch)

    @staticmethod
    def _check_config(config):
        assert isinstance(config, dict)

        assert "model" in config and "data" in config and "version" in config
        model_cfg = config["model"]
        data_cfg = config["data"]

        assert "network" in model_cfg
        assert set(model_cfg["network"].keys()) == {
            "backbone", "extractor", "main_head", "ssl_head"
        }
        assert "optimizer" in model_cfg
        assert "losses" in model_cfg


if __name__ == '__main__':
    import torch
    from ttt.constants import DATASET_DIR

    main_classes = 10
    ssl_classes = 4
    model_cfg = {
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
        },
        # define all losses
        "losses": [
            {"name": "cross-entropy", "params": {}}
        ],
        "batch_size": 32,
        "num_workers": 1,
    }
    data_cfg = {
        "root": DATASET_DIR,
        "dataset": {
            "name": "cifar_dataset",
            "params": {
                "train": {}
            },
            "config": [{"name": "CIFAR-10", "version": None, "mode": "train"}]
        },
        "signal_transform": {
            "train": [
                {
                    "name": "Permute",
                    "params": {
                        "order": [2, 0, 1]
                    }
                },
                {
                    "name": "Rescale",
                    "params": {
                        "value": 255.0
                    }
                },
                {
                    "name": "Normalize",
                    "params": {
                        "mean": "cifar",
                        "std": "cifar"
                    }
                }
            ],
            "test": [
                {
                    "name": "Permute",
                    "params": {
                        "order": [2, 0, 1]
                    }
                },
                {
                    "name": "Rescale",
                    "params": {
                        "value": 255.0
                    }
                },
                {
                    "name": "Normalize",
                    "params": {
                        "mean": "cifar",
                        "std": "cifar"
                    }
                }
            ]
        },
        "target_transform": {
            "name": "classification",
            "params": {
                "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            }
        },
        "collate_fn": {
            "name": "base",
            "params": {}
        }
    }

    config = {
        "model": model_cfg,
        "data": data_cfg,
        "version": "test"
    }

    classifier = ClassificationModel(config)

    # check networks
    assert classifier.main_net.extractor == classifier.ssl_net.extractor
    assert classifier.main_net.head != classifier.ssl_net.head

    # check optimizer and scheduler
    assert hasattr(classifier, 'optimizer')
    assert hasattr(classifier, 'scheduler')

    # check losses
    assert hasattr(classifier, "losses")
    loss_fn = classifier.losses['cross-entropy']
    predictions = torch.tensor([[0.3, 0.7]])
    targets = torch.tensor([1])
    loss = loss_fn(predictions, targets).numpy() 
    assert loss == np.array(0.5130153, dtype=np.float32)

    # check training
    train_dataloader = classifier.data_module.train_dataloader()
    test_dataloader = classifier.data_module.test_dataloader()
    classifier.fit(10, train_dataloader, test_dataloader)
