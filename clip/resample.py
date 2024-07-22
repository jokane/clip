""" Tools for resampling clips. """

import numpy as np
import scipy.signal

from .base import MutatorClip, require_clip
from .metrics import Metrics
from .validate import require_float, require_positive

class resample(MutatorClip):
    """ Change the sample rate and/or length.

    :param clip: A clip to modify.
    :param sample_rate: The desired sample rate.
    :param length: The desired length.

    Use `None` for `sample_rate` or `length` to leave that part unchanged.

    |modify|"""
    def __init__(self, clip, sample_rate=None, length=None):

        super().__init__(clip)

        if sample_rate is not None:
            require_float(sample_rate, "sample rate")
            require_positive(sample_rate, "sample rate")
        else:
            sample_rate = self.clip.sample_rate()

        if length is not None:
            require_float(length, "length")
            require_positive(length, "length")
        else:
            length = self.clip.length()

        self.metrics = Metrics(src=self.clip.metrics,
                               sample_rate=sample_rate,
                               length=length)

    def old_time(self, t):
        """ Return the time in the original clip to be used at the given
        time of the present clip. """
        assert t <= self.length()
        seconds_here = t
        seconds_there = seconds_here * self.clip.length() / self.length()
        assert seconds_there <= self.clip.length()
        return seconds_there

    def new_time(self, t):
        """ Return the time in the present clip that corresponds to the given
        time of the original clip. """
        assert t <= self.clip.length()
        seconds_there = t
        seconds_here = seconds_there * self.length() / self.clip.length()
        assert seconds_here <= self.length()
        return seconds_here

    def frame_signature(self, t):
        return self.clip.frame_signature(self.old_time(t))

    def request_frame(self, t):
        self.clip.request_frame(self.old_time(t))

    def get_frame(self, t):
        return self.clip.get_frame(self.old_time(t))

    def get_samples(self):
        data = self.clip.get_samples()
        if self.clip.sample_rate() != self.sample_rate() or self.clip.length() != self.length():
            data = scipy.signal.resample(data, self.num_samples())
        return data

    def get_subtitles(self):
        for subtitle in self.clip.get_subtitles():
            yield (self.new_time(subtitle[0]),
                   self.new_time(subtitle[1]),
                   subtitle[2])

def timewarp(clip, factor):
    """ Speed up a clip by the given factor. |modify|

    :param clip: A clip to modify.
    :param factor: A float factor by while to scale the clip's speed.

    """
    require_clip(clip, "clip")
    require_float(factor, "factor")
    require_positive(factor, "factor")

    return resample(clip, length=clip.length()/factor)

class reverse(MutatorClip):
    """ Reverse both the video and audio in a clip. |modify|

    :param clip: A clip to modify.

    """
    def frame_signature(self, t):
        return self.clip.frame_signature(self.length() - t)
    def get_frame(self, t):
        return self.clip.get_frame(self.length() - t)
    def get_samples(self):
        return np.flip(self.clip.get_samples(), axis=0)

