"""Utility functions for image operations"""
import cv2
import numpy as np
from PIL import Image


def read_img(path: str, order: str = 'RGB'):
    """
    Read image from the path as either 'BGR' or 'RGB'.

    :param path: str, image path to read
    :param order: str, choice of whether to load the image as
                  in 'BGR' format or 'RGB', default='BGR'
    :return: np.ndarray, image as 'BGR' or 'RGB' based on `order`
    """
    im = Image.open(path)
    im = np.asarray(im)

    if order == 'BGR':
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

    return im
