"""Contains functions for loading data"""
# pylint: disable=no-member
import logging
import inspect
from functools import partial
from collections import defaultdict
from typing import Tuple, Dict, List
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from ttt.data import dataset_factory
from ttt.data.image_transforms import ImageTransformer
from ttt.data.target_transforms import annotation_factory
from ttt.utils.logger import color


def get_dataloader(
        cfg: Dict, mode: str, batch_size: int,
        num_workers: int = 10, shuffle: bool = True, drop_last: bool = True
        ) -> DataLoader:
    """Creates the DataLoader

    :param cfg: config specifying the dataloader
    :type cfg: Dict
    :param mode: mode/split to load; one of {'train', 'test', 'val'}
    :type mode: str
    :param batch_size: number of instances in each batch
    :type batch_size: int
    :param num_workers: number of cpu workers to use, defaults to 10
    :type num_workers: int
    :param shuffle: whether to shuffle the data, defaults to True
    :type shuffle: bool, optional
    :param drop_last: whether to include last batch containing sample
        less than the batch size, defaults to True
    :type drop_last: bool, optional
    :returns: the DataLoader object
    """
    logging.info(color('Creating {} DataLoader'.format(mode), 'blue'))

    # define target transform
    target_transform = None
    if 'target_transform' in cfg:
        target_transform = annotation_factory.create(
            cfg['target_transform']['name'],
            **cfg['target_transform']['params'])

    # define signal transform
    signal_transform = None
    if 'signal_transform' in cfg:
        signal_transform = ImageTransformer(cfg['signal_transform'][mode])

    # define Dataset object
    dataset_params = cfg['dataset']['params'].get(mode, {})

    dataset_params.update({
        'target_transform': target_transform,
        'signal_transform': signal_transform,
        'mode': mode,
        'data_root': cfg['root'],
        'dataset_config': cfg['dataset']['config']
    })

    dataset = dataset_factory.create(cfg['dataset']['name'], **dataset_params)

    # to load entire dataset in one batch
    if batch_size == -1:
        batch_size = len(dataset)

    # return the DataLoader object
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True)

if __name__ == '__main__':
    from ttt.constants import DATASET_DIR

    # without any signal/target transforms
    cfg = {
        "root": DATASET_DIR,
        "dataset": {
            "name": "cifar_dataset",
            "params": {
                "train": {}
            },
            "config": [{"name": "CIFAR-10", "version": None, "mode": "train"}]
        }
    }
    train_dataloader = get_dataloader(cfg, mode="train", batch_size=128, num_workers=10)
    assert len(train_dataloader.dataset) // 128 == len(train_dataloader)
    dataset = train_dataloader.dataset
    instance = dataset[0]

    # with signal and target transforms
    cfg = {
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
            ]
        },
        "target_transform": {
            "name": "classification",
            "params": {
                "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            }
        }
    }
    train_dataloader = get_dataloader(cfg, mode="train", batch_size=128, num_workers=10)
    dataset = train_dataloader.dataset
    instance = dataset[0]
