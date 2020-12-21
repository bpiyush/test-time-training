"""Defines base dataset class for loading any dataset."""
import random
from abc import abstractmethod
from typing import List
from torch.utils.data import Dataset

from ttt.utils.typing import DatasetConfigDict

class BaseDataset(Dataset):
    """
    Defines the base dataset object that needs to be inherited
    by any task-specific dataset class and by other base classes
    for different modalities.

    :param data_root: directory where data versions / data reside
    :type data_root: str
    :param dataset_config: defines the config for the data to be loaded.
        The config is specified by a list of dicts, with each dict representing
        (dataset_name, dataset_version, mode [train, test, val])
    :type dataset_config: DatasetConfigDict
    :param fraction: fraction of the data to load, defaults to 1.0
    :type fraction: float
    """
    def __init__(self, data_root: str, dataset_config: List[DatasetConfigDict],
                 fraction: float = 1.0):
        self._check_args(fraction)
        self.data_root = data_root
        self.dataset_config = dataset_config

        self.load_items()
        self.load_fraction(fraction)

    @abstractmethod
    def load_items(self):
        """Load all data items into self.items as specified by self.dataset_config
        For example, for an image dataset, self.items will be a list of ttt.data.image.Image
        """
        pass

    def load_fraction(self, fraction: float, seed: int = 0):
        """Loads a given fraction of the dataset"""
        random.seed(0)
        if not hasattr(self, 'items'):
            raise ValueError('data items have not been loaded to self.items')

        if fraction < 1:
            orig_num = len(self.items)
            final_num = int(orig_num * fraction)
            random.shuffle(self.items)

            self.items = self.items[:final_num]

    @staticmethod
    def _check_args(fraction):
        if fraction < 0 or fraction > 1:
            raise ValueError("fraction should be within [0, 1]")
