import os
import argparse
from collections import OrderedDict

import torch

from .logger import get_logger
from . import utils


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Save Options')
    group.add_argument('--resume', default="", metavar='PATH', type=utils.path,
                       help='path to the checkpoint needed resume')
    group.add_argument('--load', default="", metavar='PATH', type=utils.path, help='The path to (pre-)trained model.')
    group.add_argument('--load-no-strict', default=True, action='store_false',
                       help='The keys of loaded model may not exactly match the model\'s. (May usefully for finetune)')
    return


def _strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


class Checkpoint(object):
    checkpoint = None

    def __init__(self, model, cfg=None, optimizer=None, scheduler=None, save_dir="", save_to_disk=True, logger=None):
        self.model = model
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk and bool(self.save_dir)
        if logger is None:
            logger = get_logger()
        self.logger = logger

    def _check_name(self, name: str):
        if not name.endswith('.pth'):
            name = name + '.pth'
        return os.path.join(self.save_dir, name)

    def save_checkpoint(self, name='checkpoint.pth', **kwargs):
        if not self.save_to_disk:
            return
        save_file = self._check_name(name)
        data = {"model": self.model.state_dict()}
        if self.cfg is not None:
            data["cfg"] = self.cfg
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)
        self.logger("Saving checkpoint to {}".format(save_file))

        torch.save(data, save_file)  # self.tag_last_checkpoint(save_file)

    def save_model(self, name='model.pth'):
        if not self.save_to_disk:
            return
        save_file = self._check_name(name)
        data = _strip_prefix_if_present(self.model.state_dict(), 'module.')
        self.logger("Saving model to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f=None):
        # if self.has_checkpoint():
        # override argument with existing checkpoint
        # f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            # self.logger("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger("==> Loading model from {}, strict: ".format(f, self.cfg.load_no_strict))
        checkpoint = torch.load(f, map_location=torch.device("cpu"))
        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching
        loaded_state_dict = _strip_prefix_if_present(checkpoint, prefix="module.")
        self.model.load_state_dict(loaded_state_dict, strict=self.cfg.load_no_strict)

        return checkpoint

    def resume(self, f=None):
        # if self.has_checkpoint():
        # override argument with existing checkpoint
        # f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            # self.logger("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger("Loading checkpoint from {}".format(f))
        if Checkpoint.checkpoint is not None:
            checkpoint = Checkpoint.checkpoint
            Checkpoint.checkpoint = None
        else:
            checkpoint = torch.load(f, map_location=torch.device("cpu"))

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching
        loaded_state_dict = _strip_prefix_if_present(checkpoint.pop("model"), prefix="module.")
        self.model.load_state_dict(loaded_state_dict)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        if "cfg" in checkpoint:
            checkpoint.pop("cfg")

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    @staticmethod
    def load_config(f=None):
        if f:
            Checkpoint.checkpoint = torch.load(f, map_location=torch.device("cpu"))
            if "cfg" in Checkpoint.checkpoint:
                print('Read config from checkpoint {}'.format(f))
                return Checkpoint.checkpoint.pop("cfg")
        return None
