""" Tools for simple audio creation and manipulation. """

import numpy as np

from .base import Clip, MutatorClip, AudioClip, VideoClip, require_clip
from .composite import composite, Element, AudioMode, VideoMode
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

class silence_audio(MutatorClip):
    """ Replace whatever audio we have with silence. """
    def get_samples(self):
        return np.zeros([self.metrics.num_samples(), self.metrics.num_channels])

def join(video_clip, audio_clip) -> Clip:
    """ Create a new clip that combines the video of one clip with the audio of
    another.  The length will be the length of the longer of the two."""
    require_clip(video_clip, "video clip")
    require_clip(audio_clip, "audio clip")

    assert not isinstance(video_clip, AudioClip)
    assert not isinstance(audio_clip, VideoClip)

    return composite(Element(video_clip, 0, [0,0],
                             video_mode=VideoMode.REPLACE,
                             audio_mode=AudioMode.IGNORE),
                     Element(audio_clip, 0, [0,0],
                             video_mode=VideoMode.IGNORE,
                             audio_mode=AudioMode.REPLACE))

class sine_wave(AudioClip):
    """ A sine wave with the given frequency. """
    def __init__(self, frequency, volume, length, sample_rate, num_channels):
        super().__init__()

        require_float(frequency, "frequency")
        require_positive(frequency, "frequency")
        require_float(volume, "volume")
        require_positive(volume, "volume")

        self.frequency = frequency
        self.volume = volume
        self.metrics = Metrics(Clip.default_metrics,
                               length = length,
                               sample_rate = sample_rate,
                               num_channels = num_channels)

    def get_samples(self):
        samples = np.arange(self.num_samples()) / self.sample_rate()
        samples = self.volume * np.sin(2 * np.pi * self.frequency * samples)
        samples = np.stack([samples]*self.num_channels(), axis=1)
        return samples

    def get_subtitles(self):
        return []

