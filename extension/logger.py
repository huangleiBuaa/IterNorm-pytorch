import argparse
import os
import warnings

logger = None


class _Logger:
    def __init__(self, filename=None, path='.', only_print=False, append=False):
        os.makedirs(path, exist_ok=True)
        self.filename = os.path.join(path, filename)
        self.file = None
        if filename and not only_print:
            self.file = open(self.filename, 'a' if append else 'w')

    def __del__(self):
        if self.file:
            self.file.close()

    def __call__(self, msg='', end='\n', is_print=True, is_log=True):
        if is_print:
            print(msg, end=end)
        if is_log and self.file is not None:
            self.file.write(msg)
            self.file.write(end)
            self.file.flush()


def add_arguments(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('Logger Options')
    # group.add_argument('--log', metavar='PATH', default='./results', help='The root path of save log text and model')
    group.add_argument('--log-suffix', metavar='NAME', default='', help='the suffix of log path.')
    group.add_argument('--print-f', metavar='N', default=100, type=int, help='print frequency. (default: 100)')
    return


def setting(filename=None, path='.', only_print=False, append=False):
    global logger
    logger = _Logger(filename, path, only_print, append)
    return logger


def get_logger():
    global logger
    if logger is None:
        warnings.warn('Logger is not set!')
        return print
    else:
        return logger
