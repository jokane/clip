""" Tools for manipulating the subtitles in a clip."""

import heapq

from .base import MutatorClip
from .validate import (require_iterable, require_float, require_non_negative, require_less,
                       require_less_equal, require_string)

class add_subtitles(MutatorClip):
    """ Add one or more subtitles to a clip. |modify|

    :param clip: The original clip.
    :param language: A string identifier for the language of the subtitles.
    :param args: Subtitles to add, each a `(start_time, end_time, text)` triple.

    """
    def __init__(self, clip, language, *args):
        super().__init__(clip)
        require_string(language, 'language')
        for i, subtitle in enumerate(args):
            require_iterable(subtitle, f'subtitle {i}')
            require_float(subtitle[0], f'subtitle {i} start time')
            require_non_negative(subtitle[0], f'subtitle {i} start time')
            require_less(subtitle[0], subtitle[1], f'subtitle {i} start time',
                         f'subtitle {i} end time')
            require_less_equal(subtitle[1], clip.length(), f'subtitle {i} start time',
                               'clip length')
            require_string(subtitle[2], f'subtitle {i} text')

        self.new_lang = language
        self.new_subtitles = args

    def get_subtitles(self):
        subs = self.clip.get_subtitles()
        if self.new_lang in subs:
            subs[self.new_lang] = list(heapq.merge(self.new_subtitles, subs[self.new_lang]))
        else:
            subs[self.new_lang] = list(self.new_subtitles)
        return subs

