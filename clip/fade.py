""" Tools for fading in and out. """

from abc import ABC, abstractmethod

import numpy as np

from .base import MutatorClip
from .validate import require_float, require_non_negative, require_less_equal

class fade_base(MutatorClip, ABC):
    """Fade in from or out to silent black or silent transparency."""

    def __init__(self, clip, fade_length, transparent=False):
        super().__init__(clip)
        require_float(fade_length, "fade length")
        require_non_negative(fade_length, "fade length")
        require_less_equal(fade_length, clip.length(), "fade length", "clip length")
        self.fade_length = fade_length
        self.transparent = transparent

    @abstractmethod
    def alpha(self, t):
        """ At the given time, what scaling factor should we apply?"""

    def frame_signature(self, t):
        sig = self.clip.frame_signature(t)
        alpha = self.alpha(t)
        if alpha > 254/255:
            # In this case, the scaling is too small to make any difference in
            # the output frame, given the 8-bit representation we're using.
            return sig
        elif self.transparent:
            return [f'faded to transparent by {alpha}', sig]
        else:
            return [f'faded to black by {alpha}', sig]

    def get_frame(self, t):
        alpha = self.alpha(t)
        assert alpha >= 0.0, f'Got alpha={alpha} at time {t}'
        assert alpha <= 1.0, f'Got alpha={alpha} at time {t}'
        frame = self.clip.get_frame(t).copy()
        if alpha > 254/255:
            return frame
        elif self.transparent:
            frame[:,:,3] = (frame[:,:,3]*alpha).astype(np.uint8)
            return frame.astype(np.uint8)
        else:
            return (alpha * frame).astype(np.uint8)

    @abstractmethod
    def get_samples(self):
        """ Return samples; implemented in fade_in and fade_out below."""

class fade_in(fade_base):
    """ Fade in from silent black or silent transparency. """
    def alpha(self, t):
        return min(1, t/self.fade_length)

    def get_samples(self):
        a = self.clip.get_samples().copy()
        length = int(self.fade_length * self.sample_rate())
        num_channels = self.num_channels()
        a[0:length] *= np.linspace([0.0]*num_channels, [1.0]*num_channels, length)
        return a

class fade_out(fade_base):
    """ Fade out to silent black or silent transparency. """
    def alpha(self, t):
        return min(1, (self.length()-t)/self.fade_length)

    def get_samples(self):
        a = self.clip.get_samples().copy()
        length = int(self.fade_length * self.sample_rate())
        num_channels = self.num_channels()
        a[a.shape[0]-length:a.shape[0]] *= np.linspace([1.0]*num_channels,
                                                       [0.0]*num_channels, length)
        return a
