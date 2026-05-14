"""A function to run a sanity check on a clip, checking that it is returning
the right data types and sizes and that nothing is breaking when the clip is
used."""

import pprint

import numpy as np

from .base import frame_times
from .validate import is_non_negative, is_string, require_positive

def verify(clip, frame_rate, verbose=False):
    """Call the appropriate methods to fully realize this clip, checking
    that the right sizes and formats of images are returned by
    `get_frame()`, the right length of format of audio is returned by
    `get_samples()`, and the right kinds of subtitles are returned by
    `get_subtitles()`.

    Useful for debugging and testing.

    :param frame_rate: The desired frame rate, in frames per second.
    :param verbose: Set this to `True` to get lots of diagnostic output.
    """

    clip.metrics.verify()

    require_positive(frame_rate, 'frame rate')

    for t in frame_times(clip.length(), frame_rate):
        clip.request_frame(t)

    for t in frame_times(clip.length(), frame_rate):
        sig = clip.frame_signature(t)
        if verbose:
            print(f'{t:0.2f}', end=" ")
            pprint.pprint(sig)
        assert sig is not None

        frame = clip.get_frame(t)
        assert isinstance(frame, np.ndarray), f'{type(frame)} ({frame})'
        assert frame.dtype == np.uint8
        if frame.shape != (clip.height(), clip.width(), 4):
            raise ValueError("Wrong shape of frame returned."
              f" Got {frame.shape} "
              f" Expecting {(clip.height(), clip.width(), 4)}")

    samples = clip.get_samples()
    assert samples.shape == (clip.num_samples(), clip.num_channels()), \
            f'{type(clip)} returned the wrong shape from get_samples.  ' \
            f'Got {samples.shape}; should have been {(clip.num_samples(), clip.num_channels())}'

    subtitles = clip.get_subtitles()
    for subtitle in subtitles:
        assert len(subtitle) == 3
        assert is_non_negative(subtitle[0])
        assert is_non_negative(subtitle[1])
        assert subtitle[0] < subtitle[1]
        assert is_string(subtitle[2])

