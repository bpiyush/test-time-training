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

        self.config = config
        self.data_config = self.config.data
        self.model_config = self.config.model
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

        epoch_data = {
            "loss": defaultdict(list)
        }
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
                    colored(self.config.version, "blue"),
                    colored(epoch, "blue"),
                    colored(mode.upper(), "blue"),
                    loss
                ), refresh=True
            )

            if mode == "train":
                self._update_network_params(loss)

            epoch_data['loss']['main'].append(main_loss.detach().numpy())
            epoch_data['loss']['ssl'].append(ssl_loss.detach().numpy())
            epoch_data['loss']['total'].append(loss.detach().numpy())

        for key in epoch_data['loss']:
            epoch_data['loss'][key] = np.mean(epoch_data['loss'][key])

        return epoch_data

    def fit(self, n_epochs: int, train_dataloader, test_dataloader, use_wandb):
        wandb_logs = dict()
        for epoch in range(n_epochs):
            train_data = self._process_epoch(train_dataloader, "train", epoch)
            test_data = self._process_epoch(test_dataloader, "test", epoch)

            for key in train_data['loss']:
                wandb_logs.update({f"train/{key}-loss": train_data['loss'][key]})
                wandb_logs.update({f"test/{key}-loss": test_data['loss'][key]})

            if use_wandb:
                wandb.log(wandb_logs, step=epoch)


if __name__ == '__main__':
    import os
    from os.path import join, dirname
    import wandb
    import logging
    import torch
    from ttt.constants import DATASET_DIR
    from ttt.config import Config
    from ttt.utils.logger import set_logger
    from ttt.utils.random import seed_everything

    # fix all seeds
    seed_everything()

    # through config file
    version = "defaults/base.yml"
    cfg = Config(version=version, user=None)

    # set logging
    set_logger(join(cfg.log_dir, 'train.log'))
    logging.info({"Sample argument": 10})

    # setup W&B configuration (will need to be configured for non-wadhwani users)
    os.environ['WANDB_ENTITY'] = "wadhwani"
    os.environ['WANDB_PROJECT'] = "test-time-training"
    os.environ['WANDB_DIR'] = dirname(cfg.ckpt_dir)

    run_name = version.replace('/', '_')
    wandb.init(
        name=run_name, dir=dirname(cfg.ckpt_dir), notes=cfg.description
    )
    wandb.config.update(cfg.__dict__)

    # define the classifier
    classifier = ClassificationModel(cfg)

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
    classifier.fit(cfg.model["num_epochs"], train_dataloader, test_dataloader, use_wandb=True)
