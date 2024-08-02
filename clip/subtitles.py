""" Tools for manipulating the subtitles in a clip."""

import heapq

from .base import MutatorClip
from .validate import (require_iterable, require_float, require_non_negative, require_less,
                       require_less_equal, require_string)

class add_subtitles(MutatorClip):
    """ Add one or more subtitles to a clip. |modify|

    :param clip: The original clip.
    :param args: Subtitles to add, each a `(start_time, end_time, text)` triple.
    
    """
    def __init__(self, clip, *args):
        super().__init__(clip)
        for i, subtitle in enumerate(args):
            require_iterable(subtitle, f'subtitle {i}')
            require_float(subtitle[0], f'subtitle {i} start time')
            require_non_negative(subtitle[0], f'subtitle {i} start time')
            require_less(subtitle[0], subtitle[1], f'subtitle {i} start time',
                         f'subtitle {i} end time')
            require_less_equal(subtitle[1], clip.length(), f'subtitle {i} start time',
                               'clip length')
            require_string(subtitle[2], f'subtitle {i} text')

        self.new_subtitles = args
        self.subtitles = None

    def get_subtitles(self):
        if self.subtitles is None:
            self.subtitles = list(heapq.merge(self.new_subtitles, self.clip.get_subtitles()))
        yield from self.subtitles

