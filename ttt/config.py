"""Configuration file reader."""
import os
from os.path import join, exists, dirname, realpath
from typing import Dict

from ttt.utils.io import read_yml, save_yml
from ttt.utils.typing import NoneType
from ttt.constants import ROOT

VALID_USERS = ["piyush", None]


class Config:
    """Class for storing experiment configuration

    :param version: path of the .yml file which contains the hyperparameters,
        note that this should be path starting from $PWD/../cfg/
    :type version: str
    :param user: user whose output folder contains saved config
    :type user: str, defaults to None
    """
    def __init__(self, version: str, user: str = None):
        super(Config, self).__init__()

        self._check_args(version, user)
        self.version = version
        self.user = "piyush" if user is None else user

        HOME = dirname(dirname(realpath(__file__)))
        ALL_OUT_DIR = join(ROOT, "outputs")
        USER_OUT_DIR = join(ALL_OUT_DIR, self.user)

        self.paths = {
            "HOME": HOME, "ALL_OUT_DIR": ALL_OUT_DIR, "OUT_DIR": USER_OUT_DIR
        }

        cfg_dir = join(HOME, "cfg")
        cfg_path = join(cfg_dir, version)
        cfg_subpath = version.replace('.yml', '')

        # define relevant folders/paths
        self.output_dir = join(self.paths["OUT_DIR"], cfg_subpath)
        self.cfg_save_path = join(self.output_dir, "cfg.yml")
        self.ckpt_dir = join(self.output_dir, "checkpoints")
        self.log_dir = join(self.output_dir, "logs")

        # create missing directories
        for path in [self.ckpt_dir, self.log_dir, dirname(self.cfg_save_path)]:
            os.makedirs(path, exist_ok=True)

        # load the config and update
        self.update_from_path(cfg_path)

        # save the config
        self.save()

    @staticmethod
    def _check_args(version, user):
        assert isinstance(version, str) and version.endswith(("yml", "yaml"))

        cfg_path = join(dirname(dirname(realpath(__file__))), "cfg", version)
        assert exists(cfg_path), f"Config file does not exist at {version}"

        assert isinstance(user, (str, NoneType))
        assert user in VALID_USERS, f"user={user} not found. "\
            "Please add your username to VALID_USERS in ttt/config.py"


    def __repr__(self):
        return "Config(version={}, user={})".format(self.version, self.user)

    def save(self):
        """Saves parameters"""
        save_yml(self.__dict__, self.cfg_save_path)

    def update_from_path(self, path: str):
        """Loads parameters from yml file"""
        params = read_yml(path)
        self.update_from_params(params)

    @staticmethod
    def _set_defaults(params: Dict):
        """Validates parameter values"""

        # data
        params['data']['sampler'] = params['data'].get('sampler', {})
        params['data']['dataset']['params'] = params['data']['dataset'].get(
            'params', {})

        # loss config
        losses_cfg = [{
            'name': 'cross-entropy',
            'params': {}
        }]
        params['model']['losses'] = params['model'].get('loss', losses_cfg)

        return params

    @staticmethod
    def _check_params(params: Dict):
        """Validates parameter values"""
        assert "description" in params
        assert "data" in params
        assert "model" in params

        if 'optimizer' in params['model'] and 'scheduler' in params['model']['optimizer']:
            scheduler_config = params['model']['optimizer']['scheduler']

            if scheduler_config['name'] == 'StepLR':
                assert 'value' not in scheduler_config
            if scheduler_config['name'] == 'MultiStepLR':
                assert 'value' not in scheduler_config
            elif scheduler_config['name'] == 'ReduceLRInPlateau':
                assert scheduler_config['update'] == 'epoch'
                assert 'value' in scheduler_config
            elif scheduler_config['name'] == 'CyclicLR':
                assert scheduler_config['update'] == 'batch'
                assert 'value' not in scheduler_config
            elif scheduler_config['name'] == '1cycle':
                assert scheduler_config['update'] == 'batch'
                assert 'value' not in scheduler_config

    def update_from_params(self, params: Dict):
        """Updates parameters from dict"""
        params = self._set_defaults(params)
        self._check_params(params)
        self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `config.dict['lr']`"""
        return self.__dict__


if __name__ == '__main__':
    cfg = Config(version="defaults/base.yml", user=None)
    assert hasattr(cfg, "data") and hasattr(cfg, "model")
