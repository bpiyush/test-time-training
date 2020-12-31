from os.path import join
import pickle
import yaml


class PrettySafeLoader(yaml.SafeLoader):
    """Custom loader for reading YAML files"""
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)


def read_yml(path: str, loader_type: str = 'default'):
    """Read params from a yml file.

    :param path: filepath to read
    :type path: str
    :param loader_type: type of loader used to load yml files
    :type loader: str
    :returns: dict, contents stored in `path`
    """
    assert loader_type in ['default', 'safe']
    if loader_type == 'default':
        loader = yaml.Loader
    else:
        loader = PrettySafeLoader

    with open(path, 'r') as f:
        data = yaml.load(f, Loader=loader)

    return data


def save_yml(data: dict, path: str):
    """Save params in the given yml file path.
    
    :param data: data to save
    :type data: dict
    :param path: filepath to write to
    :type path: str
    """
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_pkl(filepath):
    """
    Loads a pkl file. encoding argument exists only for python2.
    """
    return pickle.load(open(filepath, 'rb'))


def save_pkl(data, filepath):
    """Helper to save a pkl file"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
