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
import contextlib
import collections
import hashlib
import math
import os
import tempfile

import numpy as np
import progressbar

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

def is_positive(x):
    """ Can the given value be interpreted as a positive number? """
    return x>0

def is_non_negative(x):
    """ Can the given value be interpreted as a non-negative number? """
    return x>=0

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

def require(x, func, condition, name):
    """ Make sure func(x) returns a true value, and complain if not."""
    if not func(x):
        raise TypeError(f'Expected {name} to be a {condition}, but got {x} instead.')

def require_int(x, name):
    """ Raise an informative exception if x is not an integer. """
    require(x, is_int, "integer", name)

def require_float(x, name):
    """ Raise an informative exception if x is not a float. """
    require(x, is_float, "positive real number", name)

def require_positive(x, name):
    """ Raise an informative exception if x is not positive. """
    require(x, is_positive, "positive", name)

def require_non_negative(x, name):
    """ Raise an informative exception if x is not 0 or positive. """
    require(x, is_non_negative, "non-negative", name)


class Metrics:
    """ A object describing the dimensions of a Clip. """
    def __init__(self, src=None, width=None, height=None, frame_rate=None,
                 sample_rate=None, num_channels=None, length=None):
        self.width = width if width is not None else src.width
        self.height = height if height is not None else src.height
        self.frame_rate = frame_rate if frame_rate is not None else src.frame_rate
        self.sample_rate = sample_rate if sample_rate is not None else src.sample_rate
        self.num_channels = num_channels if num_channels is not None else src.num_channels
        self.length = length if length is not None else src.length
        self.verify()

    def verify(self):
        """ Make sure we have valid metrics. """
        require_int(self.width, "width")
        require_int(self.height, "height")
        require_float(self.frame_rate, "frame rate")
        require_int(self.sample_rate, "sample rate")
        require_int(self.num_channels, "number of channels")
        require_float(self.length, "length")
        require_positive(self.width, "width")
        require_positive(self.height, "height")
        require_positive(self.frame_rate, "frame rate")
        require_positive(self.sample_rate, "sample rate")
        require_positive(self.num_channels, "number of channels")
        require_positive(self.length, "length")

    def num_frames(self):
        """Length of the clip, in video frames."""
        return int(self.length * self.frame_rate)

    def num_samples(self):
        """Length of the clip, in audio samples."""
        return int(self.length * self.sample_rate)


default_metrics = Metrics(
    width = 640,
    height = 480,
    frame_rate = 30,
    sample_rate = 48000,
    num_channels = 2,
    length = 1,
)

@contextlib.contextmanager
def temporary_current_directory():
    """Create a context in which the current directory is a new temporary
    directory.  When the context ends, the current directory is restored and
    the temporary directory is vaporized."""
    previous_current_directory = os.getcwd()

    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)

        try:
            yield
        finally:
            os.chdir(previous_current_directory)

def custom_progressbar(task, steps):
    """Return a progress bar (for use as a context manager) customized for
    our purposes."""
    digits = int(math.log10(steps))+1
    widgets = [
        '|',
        f'{task:^25s}',
        ' ',
        progressbar.Bar(),
        progressbar.Percentage(),
        '| ',
        progressbar.SimpleProgress(format=f'%(value_s){digits}s/%(max_value_s){digits}s'),
        ' |',
        progressbar.ETA(
            format_not_started='',
            format_finished='%(elapsed)8s',
            format='%(eta)8s',
            format_zero='',
            format_NA=''
        ),
        '|'
    ]
    return progressbar.ProgressBar(max_value=steps, widgets=widgets)


class ClipCache:
    """An object for managing the cache of already-computed frames, audio
    segments, and other things."""
    def __init__(self):
        self.directory = '/tmp/clipcache/'
        self.cache = None

    def scan_directory(self):
        """Examine the cache directory and remember what we see there."""
        self.cache = dict()
        try:
            for cached_frame in os.listdir(self.directory):
                self.cache[os.path.join(self.directory, cached_frame)] = True
        except FileNotFoundError:
            os.mkdir(self.directory)

        counts = '; '.join(map(lambda x: f'{x[1]} {x[0]}',
          collections.Counter(map(lambda x: os.path.splitext(x)[1][1:],
          self.cache.keys())).items()))
        print(f'Found {len(self.cache)} cached items ({counts}) in {self.directory}')

    def sig_to_fname(self, sig, ext):
        """Compute the filename where something with the given signature and
        extension should live."""
        blob = hashlib.md5(str(sig).encode()).hexdigest()
        return os.path.join(self.directory, f'{blob}.{ext}')

    def lookup(self, sig, ext):
        """Determine the appropriate filename for something with the given
        signature and extension.  Return a tuple with that filename followed
        by True or False, indicating whether that file exists or not."""
        if self.cache is None: self.scan_directory()
        cached_filename = self.sig_to_fname(sig, ext)
        return (cached_filename, cached_filename in self.cache)

    def insert(self, fname):
        """Update the cache to reflect the fact that the given file exists."""
        if self.cache is None: self.scan_directory()
        self.cache[fname] = True

cache = ClipCache()

class Clip(ABC):
    """The base class for all clips.  A finite series of frames, each with
    identical height and width, meant to be played at a given rate, along with
    an audio clip of the same length."""
    def __init__(self):
        self.metrics = None

    @abstractmethod
    def frame_signature(self, index):
        """A string that uniquely describes the appearance of the given frame."""

    @abstractmethod
    def get_frame(self, index):
        """Create and return one frame of the clip."""

    @abstractmethod
    def get_samples(self):
        """Create and return the audio data for the clip."""

    def length(self):
        """Length of the clip, in seconds."""
        return self.metrics.length

    def frame_rate(self):
        """Number of frames per second."""
        return self.metrics.frame_rate

    def num_channels(self):
        """Number of channels in the clip, i.e. mono or stereo."""
        return self.metrics.num_channels

    def sample_rate(self):
        """Number of audio samples per second."""
        return self.metrics.sample_rate

    def num_frames(self):
        """Number of audio samples per second."""
        return self.metrics.num_frames()

    def num_samples(self):
        """Number of audio samples per second."""
        return self.metrics.num_samples()

    def frame_to_sample(self, index):
        """Given a frame index, compute the corresponding sample index."""
        require_int(index, 'frame index')
        require_non_negative(index, 'frame index')
        return int(index * self.sample_rate() / self.frame_rate())

    def readable_length(self):
        """Return a human-readable description of the lenth of the clip."""
        secs = self.length()
        mins, secs = divmod(secs, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f'{hours}:{mins:02}:{secs:02}'
        else:
            return f'{mins}:{secs:02}'

    def default_samples(self):
        """Return audio samples appropriate to use as a default audio.  That
        is, silence with the appropriate metrics."""
        return np.zeros([self.metrics.num_samples(), self.metrics.num_channels], np.uint8)

class solid(Clip):
    """A video clip in which each frame has the same solid color."""
    def __init__(self, width, height, frame_rate, length, color):
        super().__init__()
        assert is_color(color)
        self.metrics = Metrics(
            default_metrics,
            width=width,
            height=height,
            frame_rate=frame_rate,
            length=length
        )

        self.color = [color[2], color[1], color[0], 0]
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

    def get_samples(self):
        return self.default_samples()
