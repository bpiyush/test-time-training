"""Utility functions for image operations"""
import cv2
import numpy as np
from PIL import Image
import warnings


def read_image(path: str, mode: str = 'RGB', astype='uint8'):
    """Read an image at given location."""
    image = cv2.imread(path)
    if mode == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if (image.max() > 1.0 or image.min() < 0) and astype == 'float':
        warning = "Caution! You are reading an image with `astype='float'`"\
            " while the image is not in [0., 1.0]"
        warnings.warn(warning)

    image = image.astype(astype)

    return image

