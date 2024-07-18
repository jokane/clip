""" Tools for simple video creation and manipulation. """

import hashlib

from .base import Clip, AudioClip, VideoClip, require_clip
from .metrics import Metrics
from .validate import (require_color, require_float, require_non_negative, require_less_equal,
                       require_positive)
from .util import read_image

class solid(Clip):
    """A video clip in which each frame has the same solid color. |ex-nihilo|"""
    def __init__(self, color, width, height, length):
        super().__init__()
        require_color(color, "solid color")
        self.metrics = Metrics(Clip.default_metrics,
                               width=width,
                               height=height,
                               length=length)

        self.color = [color[2], color[1], color[0], 255]
        self.frame = None

    # Yes, this is gross.  But it avoids both code duplication and multiple
    # inheritance, so...
    frame_signature = AudioClip.frame_signature
    request_frame = AudioClip.request_frame
    get_frame = AudioClip.get_frame
    get_samples = VideoClip.get_samples
    get_subtitles = VideoClip.get_subtitles

def black(width, height, length) -> Clip:
    """ A silent solid black clip. """
    return solid([0,0,0], width, height, length)

def white(width, height, length) -> Clip:
    """ A silent white black clip. """
    return solid([255,255,255], width, height, length)

class static_frame(VideoClip):
    """Show a single image over and over, silently."""
    def __init__(self, the_frame, frame_name, length):
        super().__init__()
        try:
            height, width, depth = the_frame.shape
        except AttributeError as e:
            raise TypeError(f"Cannot not get shape of {the_frame}.") from e
        except ValueError as e:
            raise ValueError(f"Could not get width, height, and depth of {the_frame}."
              f" Shape is {the_frame.shape}.") from e
        if depth != 4:
            raise ValueError(f"Frame {the_frame} does not have 4 channels."
              f" Shape is {the_frame.shape}.")

        self.metrics = Metrics(src=Clip.default_metrics,
                               width=width,
                               height=height,
                               length=length)

        self.the_frame = the_frame.copy()
        hash_source = self.the_frame.tobytes()
        self.sig = hashlib.sha1(hash_source).hexdigest()[:7]

        self.frame_name = frame_name

    def frame_signature(self, t):
        return [ 'static_frame', {
          'name': self.frame_name,
          'sig': self.sig
        }]

    def request_frame(self, t):
        pass

    def get_frame(self, t):
        return self.the_frame

    def get_subtitles(self):
        return []

class repeat_frame(VideoClip):
    """Show the same frame, from another clip, over and over. |modify|"""

    def __init__(self, clip, when, length):
        super().__init__()
        require_clip(clip, "clip")
        require_float(when, "time")
        require_non_negative(when, "time")
        require_less_equal(when, clip.length(), "time", "clip length")
        require_float(length, "length")
        require_positive(length, "length")

        self.metrics = Metrics(src=clip.metrics,
                               length=length)
        self.clip = clip
        self.when = when

    def frame_signature(self, t):
        return self.clip.frame_signature(self.when)

    def request_frame(self, t):
        self.clip.request_frame(self.when)

    def get_frame(self, t):
        return self.clip.get_frame(self.when)

    def get_subtitles(self):
        return []


def static_image(filename, length):
    """ Show a single image loaded from a file over and over, silently. |from-source|"""
    the_frame = read_image(filename)
    assert the_frame is not None
    return static_frame(the_frame, filename, length)

