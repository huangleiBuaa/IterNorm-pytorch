import argparse
import torch.nn as nn
from .center_normalization import CenterNorm
from .group_batch_normalization import GroupBatchNorm
from .iterative_normalization import IterNorm
#from .iterative_normalization_FlexGroup import IterNorm
from .dbn import DBN, DBN2
from ..utils import str2dict


def _GroupNorm(num_features, num_groups=32, eps=1e-5, affine=True, *args, **kwargs):
    return nn.GroupNorm(num_groups, num_features, eps=eps, affine=affine)


def _LayerNorm(normalized_shape, eps=1e-5, affine=True, *args, **kwargs):
    return nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)


def _BatchNorm(num_features, dim=4, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, *args, **kwargs):
    return (nn.BatchNorm2d if dim == 4 else nn.BatchNorm1d)(num_features, eps=eps, momentum=momentum, affine=affine,
                                                            track_running_stats=track_running_stats)


def _InstanceNorm(num_features, dim=4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False, *args,
                  **kwargs):
    return (nn.InstanceNorm2d if dim == 4 else nn.InstanceNorm1d)(num_features, eps=eps, momentum=momentum,
                                                                  affine=affine,
                                                                  track_running_stats=track_running_stats)


class _config:
    norm = 'BN'
    norm_cfg = {}
    norm_methods = {'BN': _BatchNorm, 'GN': _GroupNorm, 'LN': _LayerNorm, 'IN': _InstanceNorm, 'CN': CenterNorm,
                    'None': None, 'GBN': GroupBatchNorm, 'DBN': DBN, 'DBN2': DBN2, 'ItN': IterNorm}


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Normalization Options')
    group.add_argument('--norm', default='BN', help='Use which normalization layers? {' + ', '.join(
        _config.norm_methods.keys()) + '}' + ' (defalut: {})'.format(_config.norm))
    group.add_argument('--norm-cfg', type=str2dict, default={}, metavar='DICT', help='layers config.')
    return group


def setting(cfg: argparse.Namespace):
    for key, value in vars(cfg).items():
        if key in _config.__dict__:
            setattr(_config, key, value)
    return ('_' + _config.norm) if _config.norm != 'BN' else ''


def Norm(*args, **kwargs):
    kwargs.update(_config.norm_cfg)
    if _config.norm == 'None':
        return None
    return _config.norm_methods[_config.norm](*args, **kwargs)
