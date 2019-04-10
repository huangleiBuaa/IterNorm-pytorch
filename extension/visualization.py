import argparse

import torch
import numpy as np


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Visualization Options')
    group.add_argument('--vis', action='store_true', help='Is the visualization training process?')
    group.add_argument('--vis-port', default=6006, type=int, help='The visualization port (default 6006)')
    # group.add_argument('--vis-env', default=None, help='The env name of visdom use. Default: <model_name>')
    return


class Visualization:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.viz = None
        self.env = None
        self.names = {}
        self.values = {}
        self.windows = {}
        self.cnt = {}
        self.num = {}

    def set(self, env_name, names: dict):
        if not self.cfg.vis:
            return
        try:
            import visdom
            self.env = env_name
            self.viz = visdom.Visdom(env=env_name, port=self.cfg.vis_port)
        except ImportError:
            print('You do not install visdom!!!!')
            self.cfg.vis = False
            return
        self.names = names
        self.values = {}
        self.windows = {}
        self.cnt = {}
        self.num = {}
        for name, label in self.names.items():
            self.values[name] = 0
            self.cnt[label] = 0
            self.num[label] = 0
            self.windows.setdefault(label, [])
            self.windows[label].append(name)

        for label, names in self.windows.items():
            opts = dict(title=label, legend=names, showlegend=True, # webgl=False,
                        # layoutopts={'plotly': {'legend': {'x': 0, 'y': 0}}},
                        # marginleft=0, marginright=0, margintop=10, marginbottom=0,
                        )

            zero = np.ones((1, len(names)))
            self.viz.line(zero, zero, win=label, opts=opts)

    def add_value(self, name, value):
        if not self.cfg.vis:
            return
        if isinstance(value, torch.Tensor):
            assert value.numel() == 1
            value = value.item()
        self.values[name] = value
        label = self.names[name]
        self.cnt[label] += 1
        if self.cnt[label] == len(self.windows[label]):
            y = np.array([[self.values[name] for name in self.windows[label]]])
            x = np.ones_like(y) * self.num[label]
            opts = dict(title=label, legend=self.windows[label], showlegend=True,  # webgl=False,
                        layoutopts={'plotly': {'legend': {'x': 0.05, 'y': 1}}},
                        # marginleft=0, marginright=0, margintop=10, marginbottom=0,
                        )
            self.viz.line(y, x, update='append' if self.num[label] else 'new', win=label, opts=opts)
            self.cnt[label] = 0
            self.num[label] += 1

    def clear(self, label):
        if not self.cfg.vis:
            return
        self.num[label] = 0

    def add_images(self, images, title='images', win='images', nrow=8):
        if self.cfg.vis:
            self.viz.images(images, win=win, nrow=nrow, opts={'title': title})

    def __del__(self):
        if self.viz:
            self.viz.save([self.env])


def setting(cfg: argparse.Namespace, env_name: str, names: dict):
    vis = Visualization(cfg)
    vis.set(env_name, names)
    return vis
