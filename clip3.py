"""
clip.py
--------------------------------------------------------------------------------
This is a library for manipulating and generating short video clips.  It can
read video in any format supported by ffmpeg, and provides abstractions for
cropping, superimposing, adding text, trimming, fading in and out, fading
between clips, etc.  Additional effects can be achieved by filtering the frames
through custom functions.

The basic currency is the (abstract) Clip class, which encapsulates a sequence
of identically-sized frames, each a cv2 image, meant to be played at a certain
frame rate.  Clips may be created by reading from a video file (see:
video_file), by starting from a blank video (see: black or solid), or by using
one of the other subclasses or functions to modify existing clips.
All of these methods are non-destructive: Doing, say, a crop() on a
clip with return a new, cropped clip, but will not affect the original.  Each
clip also has an audio track, which must the same length as the video part.

Possibly relevant implementation details:
- There is a "lazy evaluation" flavor to the execution.  Simply creating a
  clip object will check for some errors (for example, mismatched sizes or
  framerates) but will not actually do any work to produce the video.  The real
  rendering happens when one calls one of the save or play methods on a clip.

- To accelerate things across multiple runs, frames are cached in
  /tmp/clipcache.  This caching is done based on a string "signature" for each
  frame, which is meant to uniquely identify the visual contents of a frame.

- Some things are subclasses of Clip, whereas other parts are just functions.
  However, all of them use snake_case because it should not matter to a user
  whether it's a subclass or a function.  Same for Audio.

--------------------------------------------------------------------------------

"""


# Changes in this version:
# Externally visible:
# - Merging Clip and Audio into one monolithic Clip class.
# - Now length and num_samples can be floats.
# Internal implementation details:
# - New Metrics class.  Each Clip should have an attribute called metrics.
#       Result: fewer abstract methods, less boilerplate.

from abc import ABC, abstractmethod
import numpy as np

def is_float(x):
    """ Can the given value be interpreted as a float? """
    try:
        float(x)
        return True
    except ValueError:
        return False

def is_int(x):
    """ Can the given value be interpreted as an int? """
    return isinstance(x, int)

def is_positive_float(x):
    """ Can the given value be interpreted as a positive float? """
    return is_float(x) and x>0

def is_positive_int(x):
    """ Can the given value be interpreted as a positive int? """
    return is_int(x) and x>0

def is_color(color):
    """ Is this a color, in RGB 8-bit format? """
    if len(color) != 3: return False
    if not is_int(color[0]): return False
    if not is_int(color[1]): return False
    if not is_int(color[2]): return False
    if color[0] < 0 or color[0] > 255: return False
    if color[1] < 0 or color[1] > 255: return False
    if color[2] < 0 or color[2] > 255: return False
    return True

class Metrics:
    """ A object describing the dimensions of a Clip. """
    def __init__(self, src=None, width=None, height=None, frame_rate=None, length=None,
                 sample_rate=None, num_channels=None, num_samples=None):
        self.width = width if width is not None else src.width
        self.height = height if height is not None else src.height
        self.frame_rate = frame_rate if frame_rate is not None else src.frame_rate
        self.length = length if length is not None else src.length
        self.sample_rate = sample_rate if sample_rate is not None else src.sample_rate
        self.num_channels = num_channels if num_channels is not None else src.num_channels
        self.num_samples = num_samples if num_samples is not None else src.num_samples
        self.verify()

    def verify(self):
        """ Make sure we have valid metrics. """
        assert is_positive_int(self.width)
        assert is_positive_int(self.height)
        assert is_positive_float(self.frame_rate)
        assert is_positive_float(self.length)
        assert is_positive_int(self.sample_rate)
        assert is_positive_int(self.num_channels)
        assert is_positive_float(self.num_samples)

default_metrics = Metrics(
    width = 640,
    height = 480,
    length = 1,
    frame_rate = 30,
    sample_rate = 48000,
    num_channels = 2,
    num_samples = 1
)

class Clip(ABC):
    """The base class for all clips.  A finite series of frames, each with
    identical height and width, meant to be played at a given rate, along with
    an audio clip of the same length."""

    @abstractmethod
    def frame_signature(self, index):
        """A string that uniquely describes the appearance of the given frame."""

    @abstractmethod
    def get_frame(self, index):
        """Create and return one frame of the clip."""

class solid(Clip):
    """A video clip in which each frame has the same solid color."""
    def __init__(self, width, height, frame_rate, length, color):
        assert is_color(color)
        self.metrics = Metrics(
            default_metrics,
            width=width,
            height=height,
            frame_rate=frame_rate,
            length=length
        )
        self.color = [color[2], color[1], color[0]]
        self.frame = None

    def frame_signature(self, index):
        return ['solid', {
            'width': self.metrics.width,
            'height': self.metrics.height,
            'frame_rate': self.metrics.frame_rate,
            'length': self.metrics.length,
            'color': self.color
        }]

    def get_frame(self, index):
        if self.frame is None:
            self.frame = np.zeros([self.metrics.height, self.metrics.width, 4], np.uint8)
            self.frame[:] = self.color
        return self.frame
