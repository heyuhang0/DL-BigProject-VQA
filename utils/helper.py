import sys
import time
from typing import Any, Generator, List

import matplotlib.pyplot as plt
import numpy as np


class ProgressBar():
    """
    Keras style progress bar
    with time estimation
    and customizable content
    """

    def __init__(self) -> None:
        super().__init__()
        self.content = ''
        self.last_item = False
        self._last_width = 0

    def _render_bar(self, i: int, total: int, width: int = 30) -> str:
        total_str = str(total)
        head = f'{str(i).rjust(len(total_str))}/{total_str} ['
        finished = int(i / total * width)
        if finished >= width:
            bar = '=' * width
        else:
            bar = '=' * finished + '>' + '.' * (width - finished - 1)
        return head + bar + '] '

    def set_content(self, content: str) -> None:
        """ Set custom content displayed after progress bar """
        if content:
            self.content = ' - ' + content
            return
        self.content = ''

    def _fix_width(self, bar: str) -> str:
        last_width = self._last_width + 1
        self._last_width = len(bar)
        if len(bar) >= last_width:
            return bar
        else:
            return bar + ' ' * (last_width - len(bar))

    def __call__(self, lst: List[Any]) -> Generator[Any, None, None]:
        """ Iterate through the given list with real time progress bar """
        self.last_item = False

        time_start = time.time()

        for i, item in enumerate(lst):
            # estimate remaining time
            eta = '?'
            if i > 0:
                eta = round((time.time() - time_start) / i * (len(lst) - i))

            # update progress bar
            bar = f'{self._render_bar(i, len(lst))}- ETA: {eta}s{self.content}'
            sys.stdout.write(f'\r{self._fix_width(bar)}')
            sys.stdout.flush()
            if i == len(lst) - 1:
                self.last_item = True
            yield item

        # show total time and time per step
        total_time = time.time() - time_start
        time_per_step = round(total_time/len(lst) * 1000)
        bar = (
            f'{self._render_bar(len(lst), len(lst))}'
            f'- {round(total_time)}s {time_per_step}ms/step{self.content}')
        sys.stdout.write(f'\r{self._fix_width(bar)}\n')
        sys.stdout.flush()


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax
