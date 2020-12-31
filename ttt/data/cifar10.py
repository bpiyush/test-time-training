"""
Defines dataset object for CIFAR10 dataset.
"""
from os.path import join
from typing import Tuple, List, Union, Any
from tqdm import tqdm
from termcolor import colored
import numpy as np
import torch
from torchvision.datasets import CIFAR10 as _CIFAR10

from ttt.utils.typing import DatasetConfigDict
from ttt.data.image import Image
from ttt.data.classification import ClassificationDataset
from ttt.data.image_transforms import ImageTransformer
from ttt.data.target_transforms import ClassificationTargetTransformer


class CIFAR10(ClassificationDataset):
    """
    Dataset object for CIFAR10

    :param corruption: type of corruption-added CIFAR10
    :type corruption: str, defaults to "original"
    :param version: if corruption == "new", which data version to load
    :type version: str, defaults to "v4"
    """
    def __init__(
            self, mode: str, data_root: str, dataset_config: List[DatasetConfigDict],
            target_transform: ClassificationTargetTransformer = None,
            signal_transform: ImageTransformer = None, fraction: float = 1.0,
            corruption: str = None, corrupt_intensity: int = None
        ):
        self._check_local_args(mode, dataset_config, corruption, corrupt_intensity)
        self.mode = mode
        self.corruption = corruption
        self.corrupt_intensity = corrupt_intensity

        super(CIFAR10, self).__init__(
            data_root, dataset_config, target_transform, signal_transform, fraction
        )

    def load_items(self):
        self.items = []

        for dataset in self.dataset_config:
            name, version = dataset['name'], dataset['version']
            dataset_dir = join(self.data_root, name, "raw")

            if name == "CIFAR-10":
                _dataset = _CIFAR10(
                    root=dataset_dir, train=(self.mode == "train"), download=True
                )
                X, y = _dataset.data, _dataset.targets

            elif name == "CIFAR-10-C":
                X = np.load(join(dataset_dir, f"{self.corruption}.npy"))
                y = np.load(join(dataset_dir, "labels.npy"))

                start = 10000 * (self.corrupt_intensity - 1)
                indices = np.array(range(start, start + 10000))
                X, y = X[indices], y[indices]

            elif name == "CIFAR-10.1":
                X = np.load(join(dataset_dir, f"cifar10.1_{version}_data.npy"))
                y = np.load(join(dataset_dir, f"cifar10.1_{version}_labels.npy"))

            iterator = range(X.shape[0])
            for i in tqdm(iterator, desc=colored(f"Loading items for {name}", "yellow")):
                item = Image(image=X[i], label={"classification": [y[i]]})
                self.items.append(item)

    @staticmethod
    def _check_local_args(mode, dataset_config, corruption, corrupt_intensity):
        for dataset in dataset_config:
            name, version = dataset['name'], dataset['version']
            if name in ["CIFAR-10-C", "CIFAR-10.1"]:
                assert mode == "test", "Only mode=test allowed for CIFAR-10-C, CIFAR-10.1"

            if name == "CIFAR-10":
                assert corruption is None and corrupt_intensity is None and version is None
                assert mode in ["train", "test"], f"Dataset {name} does not val `mode=val` defined."

            if name == "CIFAR-10-C":
                assert corruption is not None and version is None and corruption in {
                    "gaussian_blur", "fog", "frost", "defocus_blur", "zoom_blur"
                }
                assert isinstance(corrupt_intensity, int) and (1 <= corrupt_intensity <= 5)
                assert mode == "test", f"Only `mode=test is acceptable for dataset {name}`"

            if name == "CIFAR-10.1":
                assert corruption is None and corrupt_intensity is None and version in {"v4", "v6"}
                assert mode == "test", f"Only `mode=test is acceptable for dataset {name}`"


class CIDAR10DatasetBuilder:
    """Builds a CIDAR10Dataset object"""
    def __call__(self, data_root: str, mode: str,
                 dataset_config: List[dict], **kwargs):
        """Builds a CIDAR10Dataset object

        :param data_root: directory where data versions reside
        :type data_root: str
        :param mode: mode/split to load; one of {'train', 'test', 'val'}
        :type mode: str
        :param dataset_config: list of dictionaries, each containing
            (name, version, mode) corresponding to a dataset
        :type dataset_config: List[dict]
        :param **kwargs: dictionary containing values corresponding to the
            arguments of the CIDAR10Dataset class
        :type **kwargs: dict
        :returns: a ClassificationDataset object
        """
        # for i, config in enumerate(dataset_config):
        #     dataset_config[i]['mode'] = mode

        kwargs['dataset_config'] = dataset_config
        kwargs['data_root'] = data_root
        kwargs['mode'] = mode
        self._instance = CIFAR10(**kwargs)
        return self._instance


if __name__ == '__main__':
    from ttt.constants import DATASETS, DATASET_DIR

    # check CIFAR-10-C
    cifar = CIFAR10(
        mode='test',
        data_root=DATASET_DIR,
        dataset_config=[{"name": "CIFAR-10-C", "version": None}],
        corruption="gaussian_blur",
        corrupt_intensity=5
    )

    # check CIFAR-10.1
    cifar = CIFAR10(
        mode='test',
        data_root=DATASET_DIR,
        dataset_config=[{"name": "CIFAR-10.1", "version": "v4"}]
    )

    # check CIFAR-10 train
    cifar = CIFAR10(
        mode='train',
        data_root=DATASET_DIR,
        dataset_config=[{"name": "CIFAR-10", "version": None}]
    )

    # check CIFAR-10 test
    cifar = CIFAR10(
        mode='test',
        data_root=DATASET_DIR,
        dataset_config=[{"name": "CIFAR-10", "version": None}]
    )
