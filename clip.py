"""
clip.py
--------------------------------------------------------------------------------
This is a library for manipulating and generating short video clips.  It can
read video in any format supported by ffmpeg, and provides abstractions for
things like cropping, superimposing, adding text, trimming, fading in and out,
fading between clips, etc.  Additional effects can be achieved by filtering the
frames through custom functions.

The basic currency is the (abstract) Clip class, which encapsulates a stream of
identically-sized frames, each a cv2 image.  Clips may be created by reading
from a video file (see: video_file), by starting from a blank video (see:
solid), or by using one of the other subclasses or functions to modify existing
clips.  All of these methods are non-destructive: Doing, say, a crop() on a
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
  whether it's a subclass or a function.

- We assume that each image is in 8-bit BGRA format.

--------------------------------------------------------------------------------

"""

# New in this version:
# - Clips do not necessarily have frame rates.  Instead, we ask for a render at
#   a given frame rate, and each get_frame() accepts a real number timestamp in
#   seconds instead of a frame index.

# - Separate caches for exploded frames and constructed frames.

# pylint: disable=too-many-lines

from abc import ABC, abstractmethod
import contextlib
from dataclasses import dataclass
import hashlib
import math
import os
import pprint
import re
import shutil
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np
import progressbar

def is_float(x):
    """ Can the given value be interpreted as a float? """
    try:
        float(x)
        return True
    except TypeError:
        return False
    except ValueError:
        return False

def is_int(x):
    """ Can the given value be interpreted as an int? """
    return isinstance(x, int)

def is_string(x):
    """ Is the given value actually a string? """
    return isinstance(x, str)

def is_positive(x):
    """ Can the given value be interpreted as a positive number? """
    return x>0

def is_even(x):
    """ Is it an even number? """
    return x%2 == 0

def is_non_negative(x):
    """ Can the given value be interpreted as a non-negative number? """
    return x>=0

def is_color(color):
    """ Is this a color, in RGB 8-bit format? """
    try:
        if len(color) != 3: return False
    except TypeError:
        return False
    if not is_int(color[0]): return False
    if not is_int(color[1]): return False
    if not is_int(color[2]): return False
    if color[0] < 0 or color[0] > 255: return False
    if color[1] < 0 or color[1] > 255: return False
    if color[2] < 0 or color[2] > 255: return False
    return True

def is_int_point(pt):
    """ Is this a 2d point with integer coordinates. """
    if len(pt) != 2: return False
    if not is_int(pt[0]): return False
    if not is_int(pt[1]): return False
    return True

def is_iterable(x):
    """ Is this a thing that can be iterated? """
    try:
        iter(x)
        return True
    except TypeError:
        return False

def require(x, func, condition, name, exception_class):
    """ Make sure func(x) returns a true value, and complain if not."""
    if not func(x):
        raise exception_class(f'Expected {name} to be a {condition}, '
                              f'but got a {type(x)} with value {x} instead.')

def require_int(x, name):
    """ Raise an informative exception if x is not an integer. """
    require(x, is_int, "integer", name, TypeError)

def require_float(x, name):
    """ Raise an informative exception if x is not a float. """
    require(x, is_float, "float", name, TypeError)

def require_string(x, name):
    """ Raise an informative exception if x is not a string. """
    require(x, is_string, "string", name, TypeError)

def require_clip(x, name):
    """ Raise an informative exception if x is not a Clip. """
    require(x, lambda x: isinstance(x, Clip), "Clip", name, TypeError)

def require_color(x, name):
    """ Raise an informative exception if x is not a color. """
    require(x, is_color, "color", name, TypeError)

def require_int_point(x, name):
    """ Raise an informative exception if x is not a integer point. """
    require(x, is_int_point, "point with integer coordinates", name, TypeError)

def require_positive(x, name):
    """ Raise an informative exception if x is not positive. """
    require(x, is_positive, "positive number", name, ValueError)

def require_even(x, name):
    """ Raise an informative exception if x is not even. """
    require(x, is_even, "even", name, ValueError)

def require_non_negative(x, name):
    """ Raise an informative exception if x is not 0 or positive. """
    require(x, is_non_negative, "non-negative", name, ValueError)

def require_equal(x, y, name):
    """ Raise an informative exception if x and y are not equal. """
    if x != y:
        raise ValueError(f'Expected {name} to be equal, but they are not.  {x} != {y}')

def require_less_equal(x, y, name1, name2):
    """ Raise an informative exception if x is not less than or equal to y. """
    if x > y:
        raise ValueError(f'Expected "{name1}" to be less than or equal to "{name2}",'
          f' but it is not. {x} > {y}')

def require_less(x, y, name1, name2):
    """ Raise an informative exception if x is greater than y. """
    if x >= y:
        raise ValueError(f'Expected "{name1}" to be less than "{name2}", '
          f'but it is not. {x} >= {y}')

def require_callable(x, name):
    """ Raise an informative exception if x is not callable. """
    require(x, callable, "callable", name, TypeError)

class FFMPEGException(Exception):
    """Raised when ffmpeg fails for some reason."""

def ffmpeg(*args, task=None, num_frames=None):
    """Run ffmpeg with the given arguments.  Optionally, maintain a progress
    bar as it goes."""

    with tempfile.NamedTemporaryFile() as stats:
        command = f"ffmpeg -y -vstats_file {stats.name} {' '.join(args)} 2> errors"
        with subprocess.Popen(command, shell=True) as proc:
            t = threading.Thread(target=proc.communicate)
            t.start()

            if task is not None:
                with custom_progressbar(task=task, steps=num_frames) as pb:
                    pb.update(0)
                    while proc.poll() is None:
                        try:
                            with open(stats.name) as f: #pragma: no cover
                                fr = int(re.findall(r'frame=\s*(\d+)\s', f.read())[-1])
                                pb.update(min(fr, num_frames-1))
                        except FileNotFoundError:
                            pass # pragma: no cover
                        except IndexError:
                            pass # pragma: no cover
                        time.sleep(1)

            t.join()

            if os.path.exists('errors'):
                shutil.copy('errors', '/tmp/ffmpeg_errors')
                with open('/tmp/ffmpeg_command', 'w') as f:
                    print(command, file=f)

            if proc.returncode != 0:
                if os.path.exists('errors'):
                    with open('errors', 'r') as f:
                        errors = f.read()
                else:
                    errors = '[no errors file found]' #pragma: no cover
                message = (
                  f'Alas, ffmpeg failed with return code {proc.returncode}.\n'
                  f'Command was: {command}\n'
                  f'Standard error was:\n{errors}'
                )
                raise FFMPEGException(message)

def flatten_args(args):
    """ Given a list of arguments, flatten one layer of lists and other
    iterables. """
    ret = []
    for x in args:
        if is_iterable(x):
            ret += x
        else:
            ret.append(x)
    return ret

@contextlib.contextmanager
def temporarily_changed_directory(directory):
    """Create a context in which the current directory has been changed to the
    given one, which should exist already.  When the context ends, change the
    current directory back."""
    previous_current_directory = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous_current_directory)


@contextlib.contextmanager
def temporary_current_directory():
    """Create a context in which the current directory is a new temporary
    directory.  When the context ends, the current directory is restored and
    the temporary directory is vaporized."""
    with tempfile.TemporaryDirectory() as td:
        with temporarily_changed_directory(td):
            try:
                yield
            finally:
                pass

def sha256sum_file(filename):
    """ Return a short hexadecmial hash of the contents of a file. """
    # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()


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

def read_image(fname):
    """Read an image from disk, make sure it has the correct RGBA uint8 format,
    and return it."""
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Trying to open {fname}, which does not exist. "
                                f"(Current working directory is {os.getcwd()}")
    frame = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    assert frame is not None
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    assert frame.shape[2] == 4, frame.shape
    assert frame.dtype == np.uint8
    return frame

def frame_times(clip_length, frame_rate):
    """ Return the timestamps at which frames should occur for a clip of the
    given length at the given frame rate.  Specifically, generate a timestamp
    at the midpoint of the time interval for each frame. """

    frame_length = 1 / frame_rate
    t = 0.5 * frame_length

    while t <= clip_length:
        yield t
        t += frame_length

@dataclass
class Metrics:
    """ A object describing the dimensions of a Clip. """
    width: int
    height: int
    sample_rate: int
    num_channels: int
    length: float

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
        """ Make sure we have valid metrics. """
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
        """ Make sure two Metrics objects match each other.  Complain if not. """
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
        """Length of the clip, in audio samples."""
        return int(self.length * self.sample_rate)

    def readable_length(self):
        """A human-readable description of the length."""
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

class Clip(ABC):
    """The base class for all clips.  A finite series of frames, each with
    identical height and width, meant to be played at a given rate, along with
    an audio clip of the same length."""

    def __init__(self):
        self.metrics = None

    @abstractmethod
    def frame_signature(self, t):
        """A string that uniquely describes the appearance of this clip at the
        given time."""

    @abstractmethod
    def get_frame(self, t):
        """Create and return a frame of this clip at the given time."""

    @abstractmethod
    def get_samples(self):
        """Create and return the audio data for the clip."""

    # Default metrics to use when not otherwise specified.  These can make code
    # a little cleaner in a lot of places.  For example, many silent clips will
    # use the default sample rate for their dummy audio. """
    default_metrics = Metrics(width = 640,
                              height = 480,
                              sample_rate = 48000,
                              num_channels = 2,
                              length = 1)

    def length(self):
        """Length of the clip, in seconds."""
        return self.metrics.length

    def width(self):
        """Width of the video, in pixels."""
        return self.metrics.width

    def height(self):
        """Height of the video, in pixels."""
        return self.metrics.height

    def num_channels(self):
        """Number of channels in the clip, i.e. mono or stereo."""
        return self.metrics.num_channels

    def sample_rate(self):
        """Number of audio samples per second."""
        return self.metrics.sample_rate

    def num_samples(self):
        """Number of audio samples in total."""
        return self.metrics.num_samples()

    def readable_length(self):
        """A human-readable description of the length."""
        return self.metrics.readable_length()

    def verify(self, frame_rate, verbose=False):
        """ Fully realize a clip, ensuring that no exceptions occur and that
        the right sizes of video frames and audio samples are returned. Useful
        for testing. """

        self.metrics.verify()

        require_positive(frame_rate, 'frame rate')

        for t in frame_times(self.length(), frame_rate):
            sig = self.frame_signature(t)
            if verbose:
                pprint.pprint(sig)
                print(t, end=" ")
            assert sig is not None

            frame = self.get_frame(t)
            assert isinstance(frame, np.ndarray), f'{type(frame)} ({frame})'
            assert frame.dtype == np.uint8
            if frame.shape != (self.height(), self.width(), 4):
                raise ValueError("Wrong shape of frame returned."
                  f" Got {frame.shape} "
                  f" Expecting {(self.height(), self.width(), 4)}")

        samples = self.get_samples()
        assert samples.shape == (self.num_samples(), self.num_channels())



class VideoClip(Clip):
    """ Inherit from this for Clip classes that really only have video, to
    default to silent audio. """
    def get_samples(self):
        """Return audio samples appropriate to use as a default audio.  That
        is, silence with the appropriate metrics."""
        return np.zeros([self.metrics.num_samples(), self.metrics.num_channels])


class AudioClip(Clip):
    """ Inherit from this for Clip classes that only really have audio, to
    default to simple black frames for the video. """
    def __init__(self):
        super().__init__()
        self.color = [0, 0, 0, 255]
        self.frame = None

    def frame_signature(self, t):
        return ['solid', {
            'width': self.metrics.width,
            'height': self.metrics.height,
            'color': self.color
        }]

    def get_frame(self, t):
        if self.frame is None:
            self.frame = np.zeros([self.metrics.height, self.metrics.width, 4], np.uint8)
            self.frame[:] = self.color
        return self.frame

class MutatorClip(Clip):
    """ Inherit from this for Clip classes that modify another clip.
    Override only the parts that need to change."""
    def __init__(self, clip):
        super().__init__()
        require_clip(clip, "base clip")
        self.clip = clip
        self.metrics = clip.metrics

    def frame_signature(self, t):
        return self.clip.frame_signature(t)

    def get_frame(self, t):
        return self.clip.get_frame(t)

    def get_samples(self):
        return self.clip.get_samples()

class solid(Clip):
    """A video clip in which each frame has the same solid color."""
    def __init__(self, color, width, height, length):
        super().__init__()
        require_color(color, "solid color")
        self.metrics = Metrics(Clip.default_metrics,
                               width=width,
                               height=height,
                               length=length)

        self.color = [color[2], color[1], color[0], 255]
        self.frame = None

    # Avoiding both code duplication and multiple inheritance here...
    frame_signature = AudioClip.frame_signature
    get_frame = AudioClip.get_frame
    get_samples = VideoClip.get_samples

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

class scale_alpha(MutatorClip):
    """ Scale the alpha channel of a given clip by the given factor, which may
    be a float (for a constant factor) or a float-returning function (for a
    factor that changes across time)."""
    def __init__(self, clip, factor):
        super().__init__(clip)

        # Make sure we got either a constant float or a callable.
        if is_float(factor):
            factor = lambda x: x
        require_callable(factor, "factor function")

        self.factor = factor

    def frame_signature(self, t):
        factor = self.factor(t)
        require_float(factor, f'factor at time {t}')
        return ['scale_alpha', self.clip.frame_signature(t), factor]

    def get_frame(self, t):
        factor = self.factor(t)
        require_float(factor, f'factor at time {t}')
        frame = self.clip.get_frame(t)
        if factor != 1.0:
            frame = frame.astype('float')
            frame[:,:,3] *= factor
            frame = frame.astype('uint8')
        return frame

def black(width, height, length):
    """ A silent solid black clip. """
    return solid([0,0,0], width, height, length)

def white(width, height, length):
    """ A silent white black clip. """
    return solid([255,255,255], width, height, length)


