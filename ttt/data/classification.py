"""
Defines ClassificationDataset class which is used for classification tasks
where each input has a single output.
"""
from os.path import join
from abc import abstractmethod
from typing import Tuple, List, Union, Any

import torch

from ttt.data.base import BaseDataset
from ttt.data.image_transforms import ImageTransformer
from ttt.data.target_transforms import ClassificationTargetTransformer
from ttt.utils.typing import DatasetConfigDict
from ttt.factory import Factory


class ClassificationDataset(BaseDataset):
    """Dataset class for single-label classification
    :param data_root: directory where data versions reside
    :type data_root: str
    :param dataset_config: defines the config for the
        data to be loaded. The config is specified by a list of dict, with
        each dict representing: (dataset_name, dataset_version,
        mode [train, test, val])
    :type dataset_config: DatasetConfigDict
    :param target_transform: defines the transformation
        to be applied on the raw targets to make them processable.
    :type target_transform: ClassificationTargetTransformer
    :param signal_transform: defines the list of transformations to be
        applied on the raw signals where each transform is defined by a
        TransformDict object
    :type signal_transform: ImageTransformer
    :param fraction: fraction of the data to load, defaults to 1.0
    :type fraction: float
    """
    def __init__(
            self, data_root: str, dataset_config: List[DatasetConfigDict],
            target_transform: ClassificationTargetTransformer = None,
            signal_transform: ImageTransformer = None, fraction: float = 1.0):
        super(ClassificationDataset, self).__init__(
            data_root, dataset_config, fraction
        )
        self.target_transform = target_transform
        self.signal_transform = signal_transform

    @abstractmethod
    def load_items(self):
        pass

    def __len__(self):
        return len(self.items)

    def __getitem__(
            self, index: int, as_tensor=True
            ) -> Tuple[torch.Tensor, Union[List[str], int]]:
        item = self.items[index]
        self._check_item(item)

        signal = item.load(as_tensor=as_tensor)['signal']

        if self.signal_transform is not None:
            signal = self.signal_transform(signal)

        label = item.label['classification']
        if self.target_transform is not None:
            label = self.target_transform(label)

        instance = {
            'signal': signal,
            'label': label,
            'item': item
        }

        return instance


    def _check_item(self, item: Any):
        assert 'classification' in item.label,\
            "Item at index {} has no 'classification' in label".format(
                index)


class ClassificationDatasetBuilder:
    """Builds a ClassificationDataset object"""
    def __call__(self, data_root: str, mode: str,
                 dataset_config: List[dict], **kwargs):
        """Builds a ClassificationDataset object

        :param data_root: directory where data versions reside
        :type data_root: str
        :param mode: mode/split to load; one of {'train', 'test', 'val'}
        :type mode: str
        :param dataset_config: list of dictionaries, each containing
            (name, version, mode) corresponding to a dataset
        :type dataset_config: List[dict]
        :param **kwargs: dictionary containing values corresponding to the
            arguments of the ClassificationDataset class
        :type **kwargs: dict
        :returns: a ClassificationDataset object
        """
        for i, config in enumerate(dataset_config):
            dataset_config[i]['mode'] = mode

        kwargs['dataset_config'] = dataset_config
        kwargs['data_root'] = data_root
        self._instance = ClassificationDataset(**kwargs)
        return self._instance
