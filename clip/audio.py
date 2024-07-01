""" Tools for simple audio manipulations. """

import numpy as np

from .base import MutatorClip
from .metrics import Metrics
from .validate import require_float, require_positive

class mono_to_stereo(MutatorClip):
    """ Change the number of channels from one to two. """
    def __init__(self, clip):
        super().__init__(clip)
        if self.clip.metrics.num_channels != 1:
            raise ValueError(f"Expected 1 audio channel, not {self.clip.num_channels()}.")
        self.metrics = Metrics(self.metrics, num_channels=2)
    def get_samples(self):
        data = self.clip.get_samples()
        return np.concatenate((data, data), axis=1)

class stereo_to_mono(MutatorClip):
    """ Change the number of channels from two to one. """
    def __init__(self, clip):
        super().__init__(clip)
        if self.clip.metrics.num_channels != 2:
            raise ValueError(f"Expected 2 audio channels, not {self.clip.num_channels()}.")
        self.metrics = Metrics(self.metrics, num_channels=1)
    def get_samples(self):
        data = self.clip.get_samples()
        return (0.5*data[:,0] + 0.5*data[:,1]).reshape(self.num_samples(), 1)

class scale_volume(MutatorClip):
    """ Scale the volume of audio in a clip.  """
    def __init__(self, clip, factor):
        super().__init__(clip)
        require_float(factor, "scaling factor")
        require_positive(factor, "scaling factor")
        self.factor = factor

    def get_samples(self):
        return self.factor * self.clip.get_samples()
