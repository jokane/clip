""" A tool for creating video from an glob of filenames. """

import glob
import os

import cv2

from .base import Clip, VideoClip, FiniteIndexed
from .metrics import Metrics
from .validate import require_string
from .util import read_image

class image_glob(VideoClip, FiniteIndexed):
    """Video from a collection of identically-sized image files that match a
    unix-style pattern, at a given frame rate or timed to a given length.
    |from-source|

    :param pattern: A wildcard pattern, of the form used by the standard
            library function `glob.glob`, identifying a set of image files.
            These will be the frames of the clip, in sorted order.
    :param frame_rate: The clip's frame rate, in frames per second.
    :param length: The clip's length, in seconds.

    Exactly one of `num_frames` and `frame_rate` should be given; the other
    should be `None` and will be computed to match.

    """

    def __init__(self, pattern, frame_rate=None, length=None):
        VideoClip.__init__(self)

        require_string(pattern, "pattern")

        self.pattern = pattern

        self.filenames = sorted(glob.glob(pattern))
        if len(self.filenames) == 0:
            raise FileNotFoundError(f'No files matched pattern: {pattern}')

        # Get full pathnames, in case the current directory changes.
        self.filenames = list(map(lambda x: os.path.join(os.getcwd(), x), self.filenames))
        FiniteIndexed.__init__(self, len(self.filenames), frame_rate, length)

        sample_frame = cv2.imread(self.filenames[0])
        assert sample_frame is not None

        self.metrics = Metrics(src=Clip.default_metrics,
                               width=sample_frame.shape[1],
                               height=sample_frame.shape[0],
                               length = len(self.filenames)/self.frame_rate)

    def frame_signature(self, t):
        return self.filenames[self.time_to_frame_index(t)]

    def request_frame(self, t):
        pass

    def get_frame(self, t):
        return read_image(self.filenames[self.time_to_frame_index(t)])

