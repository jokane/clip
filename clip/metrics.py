""" A class to represent the metrics (i.e. width, height, length, etc.) of a
clip. """

# pylint: disable=wildcard-import

from dataclasses import dataclass

from .validate import *

@dataclass
class Metrics:
    """A object describing the dimensions of a Clip.

    :param src: Another `Metrics` object, from which to draw defaults when
            other parameters below are omitted.  One reasonable choice is
            :attr:`Clip.default_metrics`.  Another is to use the metrics of
            another clip.  If this is `None`, then all of the other parameters
            must be given.
    :param width: The width of the clip, in pixels.  A positive integer.
    :param height: The height of the clip, in pixels.  A positive integer.
    :param sample_rate: The sample rate for the audio of the clip, in samples
            per second.  A positive integer.
    :param num_channels: The number of audio channels.  A positive integer,
            usually `1` or `2`.
    :param length: The length of the clip, in seconds.  A positive float.


    """

    def __init__(self, src=None, width=None, height=None,
                 sample_rate=None, num_channels=None, length=None):
        assert src is None or isinstance(src, Metrics), f'src should be Metrics, not {type(src)}'
        self.width = width if width is not None else src.width
        self.height = height if height is not None else src.height
        self.sample_rate = sample_rate if sample_rate is not None else src.sample_rate
        self.num_channels = num_channels if num_channels is not None else src.num_channels
        self.length = length if length is not None else src.length
        self.verify()

    def verify(self):
        """Make sure we have valid metrics.  If not, raise either `TypeError` or
        `ValueError` depending on what's wrong."""
        require_int(self.width, "width")
        require_int(self.height, "height")
        require_int(self.sample_rate, "sample rate")
        require_int(self.num_channels, "number of channels")
        require_float(self.length, "length")
        require_positive(self.width, "width")
        require_positive(self.height, "height")
        require_positive(self.sample_rate, "sample rate")
        require_positive(self.num_channels, "number of channels")
        require_positive(self.length, "length")

    def verify_compatible_with(self, other, check_video=True, check_audio=True, check_length=False):
        """Make sure two Metrics objects match each other.  Raise an exception if not.

        :param other: Another :class:`Metrics` object to compare to this one.
        :param check_video: Set this to `False` to ignore differences in the frame sizes.
        :param check_audio: Set this to `False` to ignore differences in audio
                sample rate and number of channels.
        :param check_length: Set this to `False` to ignore differences in the clip lengths
        """
        assert isinstance(other, Metrics)

        if check_video:
            require_equal(self.width, other.width, "widths")
            require_equal(self.height, other.height, "heights")

        if check_audio:
            require_equal(self.num_channels, other.num_channels, "numbers of channels")
            require_equal(self.sample_rate, other.sample_rate, "sample rates")

        if check_length:
            require_equal(self.length, other.length, "lengths")

    def num_samples(self):
        """:return: The length of the clip, in audio samples."""
        return int(self.length * self.sample_rate)

    def readable_length(self):
        """:return: A human-readable description of the length."""
        secs = self.length
        mins, secs = divmod(secs, 60)
        hours, mins = divmod(mins, 60)
        mins = int(mins)
        hours = int(hours)
        secs = int(secs)
        if hours > 0:
            return f'{hours}:{mins:02}:{secs:02}'
        else:
            return f'{mins}:{secs:02}'
