""" Tools for simple audio creation and manipulation. """

import numpy as np

from .base import Clip, MutatorClip, AudioClip, VideoClip, require_clip
from .composite import composite, Element, AudioMode, VideoMode
from .metrics import Metrics
from .validate import require_float, require_positive

class mono_to_stereo(MutatorClip):
    """ Change the number of channels from one to two. |modify|

    :param clip: A clip to modify, having exactly one audio channel.
    :return: A new clip, the same as the original, but with the audio channel
            duplicated.
    """
    def __init__(self, clip):
        super().__init__(clip)
        if self.clip.metrics.num_channels != 1:
            raise ValueError(f"Expected 1 audio channel, not {self.clip.num_channels()}.")
        self.metrics = Metrics(self.metrics, num_channels=2)
    def get_samples(self):
        data = self.clip.get_samples()
        return np.concatenate((data, data), axis=1)

class stereo_to_mono(MutatorClip):
    """ Change the number of channels from two to one. |modify|

    :param clip: A clip to modify, having exactly two audio channels.
    :return: A new clip, the same as the original, but with the two audio
            channels averaged into just one.

    """
    def __init__(self, clip):
        super().__init__(clip)
        if self.clip.metrics.num_channels != 2:
            raise ValueError(f"Expected 2 audio channels, not {self.clip.num_channels()}.")
        self.metrics = Metrics(self.metrics, num_channels=1)
    def get_samples(self):
        data = self.clip.get_samples()
        return (0.5*data[:,0] + 0.5*data[:,1]).reshape(self.num_samples(), 1)

class scale_volume(MutatorClip):
    """ Scale the volume of audio in a clip. |modify|

    :param clip: A clip to modify.
    :param factor: A float.
    :return: A new clip, the same as the original, but with each of its audio
            sample multiplied by `factor`.

    """
    def __init__(self, clip, factor):
        super().__init__(clip)
        require_float(factor, "scaling factor")
        require_positive(factor, "scaling factor")
        self.factor = factor

    def get_samples(self):
        return self.factor * self.clip.get_samples()

class silence_audio(MutatorClip):
    """ Replace whatever audio we have with silence. |modify|

    :param clip: A clip to modify.
    :return: A new clip, the same as the original, but with silent audio.

    The sample rate and number of channels remain unchanged.

    """
    def get_samples(self):
        return np.zeros([self.metrics.num_samples(), self.metrics.num_channels])

def join(video_clip, audio_clip):
    """ A new clip that combines the video of one clip with the audio of
    another. |modify|

    :param video_clip: A clip whose video you care about.
    :param audio_clip: A clip whose audio you care about.
    :return: A clip with the video from `video_clip` and the audio from
            `audio_clip`.

            The length of the result will be the length of the
            longer of the two inputs.

                - If `video_clip` is longer than `audio_clip`, the result will be
                  padded with silence at the end.

                - If `audio_clip` is longer than `video_clip`, the result will be
                  padded with black frames that the end.

    """
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

class from_audio_samples(AudioClip):
    """An audio clip formed formed from a given array of samples. |from-source|

    :param samples: The audio data.  A numpy array with shape `(num_samples, num_channels)`.
    :param sample_rate: The sample rate in samples per second.  A positive integer.

    The number of channels is determined by the `shape` of the given samples
    array. The length of the clip is computed from the `shape` and the sample
    rate.

    """

    def __init__(self, samples, sample_rate):
        super().__init__()
        require_float(sample_rate, "sample rate")
        require_positive(sample_rate, "sample rate")

        self.samples = samples

        self.metrics = Metrics(Clip.default_metrics,
                               length=len(self.samples)/sample_rate,
                               sample_rate=sample_rate,
                               num_channels=self.samples.shape[1])

    def get_samples(self):
        return self.samples

    def get_subtitles(self):
        return []


class sine_wave(AudioClip):
    """ A sine wave with the given frequency. |ex-nihilo|

    :param frequency: The desired frequency, in hertz.
    :param volume: The desired volume, between 0 and 1.
    :param length: The desired length, in seconds.
    :param num_channels: The number of channels, usually `1` or `2`.

    """
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

