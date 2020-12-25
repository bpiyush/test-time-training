"""
Defines util functions on networks.

Some ideas borrowed from 
https://github.com/yueatsprograms/ttt_cifar_release/blob/master/models/SSHead.py
"""
import numpy as np
from torch import nn
import math
import copy


class ViewFlatten(nn.Module):
    """
    Flattening layer.
    """
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def unique(array):
    """np.unique without sorting"""
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]


def get_layer_names(network: nn.Module):
    """Returns layer names of each layer in network"""
    layers = []
    for name, layer in network.named_modules():
        layer = name.split(".")[0]
        if len(layer):
            layers.append(layer)

    return unique(layers)


def get_extractor_from_network(
    network: nn.Module, layer_name: str, add_flat: bool = True
    ):
    layer_names = get_layer_names(network)
    layer_names = list(layer_names)
    assert layer_name in layer_names

    index = layer_names.index(layer_name)
    ext_layer_names = layer_names[:index + 1]
    ext_layers = []
    for layer_name in ext_layer_names:
        ext_layers.append(getattr(network, layer_name))

    if add_flat:
        ext_layers.append(ViewFlatten())
    ext_layers = copy.deepcopy(ext_layers)
    
    extractor = nn.Sequential(*ext_layers)

    return extractor


if __name__ == '__main__':
    import torch
    from ttt.models.networks.resnet import ResNetCIFAR as ResNet

    network = ResNet(depth=26, width=1, classes=10)
    x = torch.randn((1, 3, 32, 32))

    extractor = get_extractor_from_network(network, 'layer2', add_flat=True)
    y = extractor(x)
    assert y.shape == (1, 8192)

    extractor = get_extractor_from_network(network, 'layer2', add_flat=False)
    y = extractor(x)
    assert y.shape == (1, 32, 16, 16)

    extractor = get_extractor_from_network(network, 'avgpool', add_flat=True)
    y = extractor(x)
    assert y.shape == (1, 64)

    extractor = get_extractor_from_network(network, 'avgpool', add_flat=False)
    y = extractor(x)
    assert y.shape == (1, 64, 1, 1)
