"""Defines the Image class which is the cannonical object of every image dataset"""
from typing import Union, List, Tuple
from os.path import exists
import numpy as np
import torch
from ttt.utils.typing import LabelDict
from ttt.utils.image import read_img


class Image:
    """Defines an image containing annotations"""
    def __init__(self, image: np.ndarray = None, path: str = None, label: LabelDict = None):
        """Constructor for the class

        :param image: image values (H x W x C)
        :type image: np.ndarray, defaults to None
        :param path: path to image
        :type path: str, defaults to None
        :param label: dictionary of labels for different tasks,
            defaults to None
        :type label: LabelDict, optional
        """
        self._check_args(image, path, label)
        self.image = image
        self.path = path
        self.label = label

    def load(
            self, mode: str = 'RGB', as_tensor: bool = False
            ) -> Union[np.ndarray, torch.Tensor]:
        """Read the image

        :param mode: 'RGB' or 'BGR', defaults to 'RGB'
        :type mode: str, optional
        :param as_tensor: whether to return a `torch.Tensor` object; returns
            a np.ndarray object if `False`, defaults to `False`
        :type as_tensor: bool, optional
        :returns: image either as a nunpy array or torch.Tensor
        """
        if self.image is None:
            self.image = read_img(self.path, mode)

        self.height, self.width = self.image.shape[:2]

        if as_tensor:
            self.image = torch.from_numpy(self.image.copy()).float()

        return {
            'signal': self.image
        }
    
    @staticmethod
    def _check_args(image, path, label):
        assert (image is not None) or (path is not None), \
            "Both `image` and `path` cannot be None"

        # check image is appropriate sized np.ndarray
        if image is not None:
            assert isinstance(image, np.ndarray)
            assert len(image.shape) == 3 and image.shape[-1] in {1, 3}
            assert path is None, f"Cannot pass path={path} when image is not None"

        # check path is a string and it exists
        if path is not None:
            assert isinstance(path, str) and exists(path)
            assert image is None, f"Cannot pass image={image} when path is not None"

        # ensure that label is of the right data type
        if label is not None:
            assert isinstance(label, dict)


if __name__ == '__main__':
    # testing for a sample image
    import numpy as np
    from os.path import join

    ROOT = "/scratch/users/piyushb/test-time-training/"
    DATASET = join(ROOT, "datasets/CIFAR-10.1/")
    cifar_path = join(DATASET, "raw/cifar10.1_v4_data.npy")

    # loading image by passing an image
    sample_image = np.load(cifar_path)[0]
    image = Image(image=sample_image)

    # loading image by passing a path
    sample_image_path = join(DATASET, "raw/sample.png")
    image = Image(path=sample_image_path)
    import ipdb; ipdb.set_trace()
