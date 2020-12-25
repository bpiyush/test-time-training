"""
Defines util functions on networks.

Some ideas borrowed from 
https://github.com/yueatsprograms/ttt_cifar_release/blob/master/models/SSHead.py
"""
import numpy as np
import torch
from torch import nn
import math
import copy
import logging
from collections import OrderedDict

from ttt.models.networks.layers import ViewFlatten, layer_factory


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


def get_head_from_network(
    network: nn.Module, layer_name: str, add_layers: list = []
    ):
    layer_names = get_layer_names(network)
    layer_names = list(layer_names)
    assert layer_name in layer_names

    index = layer_names.index(layer_name)
    head_layer_names = layer_names[index + 1:-1]
    head_layers = []
    for layer_name in head_layer_names:
        head_layers.append(getattr(network, layer_name))

    for index, layer_config in enumerate(add_layers):
        assert set(layer_config.keys()) == {'name', 'params'}
        layer = layer_factory.create(
            layer_config['name'], **layer_config['params']
        )
        head_layers.append(layer)    
    head_layers = copy.deepcopy(head_layers)

    head = nn.Sequential(*head_layers)

    return head


def _correct_state_dict(
        loaded_state_dict: OrderedDict,
        model_state_dict: OrderedDict
    ) -> OrderedDict:
    """Only retains key from the `loaded_state_dict` that match with `model_state_dict`"""
    corrected_state_dict = OrderedDict()
    for key, value in loaded_state_dict.items():
        if key not in model_state_dict or value.shape != model_state_dict[key].shape:
            logging.info(f'Removing {key} from state_dict')
            continue

        corrected_state_dict[key] = value
    return corrected_state_dict


if __name__ == '__main__':
    import torch
    from ttt.models.networks.resnet import ResNetCIFAR as ResNet

    network = ResNet(depth=26, width=1, classes=10)
    x = torch.randn((1, 3, 32, 32))

    # extractor from layer 2
    extractor = get_extractor_from_network(network, 'layer2', add_flat=True)
    y = extractor(x)
    assert y.shape == (1, 8192)

    extractor = get_extractor_from_network(network, 'layer2', add_flat=False)
    y = extractor(x)
    assert y.shape == (1, 32, 16, 16)

    # extractor from layer3 (+ until avgpool)
    extractor = get_extractor_from_network(network, 'avgpool', add_flat=True)
    y = extractor(x)
    assert y.shape == (1, 64)

    extractor = get_extractor_from_network(network, 'avgpool', add_flat=False)
    y = extractor(x)
    assert y.shape == (1, 64, 1, 1)

    # head from layer 2
    num_classes = 4
    add_layers = [
        {"name": "ViewFlatten", "params": {}},
        {"name": "Linear", "params": {"in_features": 64 * 1, "out_features": num_classes}}
    ]
    head = get_head_from_network(network, 'layer2', add_layers=add_layers)
    z = torch.randn((1, 32, 16, 16))
    y = head(z)
    assert y.shape == (1, num_classes)

    # head from layer 3
    # NOTE: you only need ViewFlatten layer when you have passed add_flat=False
    # in creating extractor from layer3
    num_classes = 4
    add_layers = [
        {"name": "Linear", "params": {"in_features": 64 * 1, "out_features": num_classes}}
    ]
    head = get_head_from_network(network, 'avgpool', add_layers=add_layers)
    z = torch.randn((1, 64))
    y = head(z)
    assert y.shape == (1, num_classes)
