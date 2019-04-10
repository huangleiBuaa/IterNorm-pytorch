# coding:utf-8
import os
import sys
import time

__all__ = ['ProgressBar', 'format_time']


def _get_terminal_size():
    try:
        columns, lines = os.get_terminal_size()
        return int(columns)
    except OSError:
        return -1


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class ProgressBar(object):
    def __init__(self, total=100, max_length=160):
        self.start_time = time.time()
        self.iter = 0
        self.total = total
        self.max_length = max_length
        self.msg_on_bar = ''
        self.msg_end = ''
        self.bar_length = 80

    def reset(self):
        self.start_time = time.time()
        self.iter = 0

    def _deal_message(self):
        msg = self.msg_on_bar.strip().lstrip()
        if len(msg) > self.bar_length:
            msg = msg[0:self.bar_length - 3]
            msg += '...'
        # center message
        msg = ' ' * ((self.bar_length - len(msg)) // 2) + msg
        msg = msg + ' ' * (self.bar_length - len(msg))
        self.msg_on_bar = msg

    def _raw_output(self):
        self.bar_length = 50
        show_len = int(self.iter / self.total * self.bar_length)
        msg = '\r|' + '>' * show_len + ' ' * (self.bar_length - show_len)
        msg += '|  ' + self.msg_end + '  ' + self.msg_on_bar
        sys.stdout.write(msg)

    def step(self, msg='', add=1):
        """
        :param add: How many iterations are executed?
        :param msg: the message need to be shown on the progress bar
        """
        self.iter = min(self.iter + add, self.total)
        if not isinstance(msg, str):
            msg = '{}'.format(msg)
        self.msg_end = ' {}/{}'.format(self.iter, self.total)
        used_time = time.time() - self.start_time
        self.msg_end += ' {}'.format(format_time(used_time))
        if self.iter != self.total:
            left_time = used_time / self.iter * (self.total - self.iter)
            self.msg_end += '<={}'.format(format_time(left_time))
        self.msg_on_bar = msg

        columns = min(_get_terminal_size(), self.max_length)
        if columns < 0:
            self._raw_output()
        else:
            self.bar_length = columns - len(self.msg_end)
            self._linux_output()

        if self.iter == self.total:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def time_used(self):
        used_time = time.time() - self.start_time
        return format_time(used_time)

    def _linux_output(self):
        show_len = int(self.iter / self.total * self.bar_length)
        self._deal_message()

        control = '\r'  # 回到行首
        control += '\33[4m'  # 下划线
        control += '\33[40;37m'  # 黑底白字
        # control += '\33[7m'         # 反显
        # control += '\33[?25l'  # 隐藏光标
        control += self.msg_on_bar[0:show_len]
        # control += '\33[0m'         # 反显
        control += '\33[47;30m'  # 白底黑字
        control += self.msg_on_bar[show_len:self.bar_length]
        # control += '\33[K'  # 清除从光标到行尾的内容
        control += '\33[0m'

        sys.stdout.write(control)
        sys.stdout.write(self.msg_end)


if __name__ == '__main__':
    bar = ProgressBar()

    epoch = 0
    while True:
        bar.reset()
        for i in range(bar.total):
            bar.step(epoch)
            time.sleep(0.1)
        epoch += 1
