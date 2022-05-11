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
import collections
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
from PIL import ImageFont
import soundfile

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

def get_font(font, size):
    """
    Return a TrueType font for use on Pillow images, with caching to
    prevent loading the same font again and again.    (The performance
    improvement seems to be small but non-zero.)
    """
    if (font, size) not in get_font.cache:
        try:
            get_font.cache[(font, size)] = ImageFont.truetype(font, size)
        except OSError as e:
            raise ValueError(f"Failed to open font {font}.") from e
    return get_font.cache[(font, size)]
get_font.cache = {}



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

class ClipCache:
    """An object for managing the cache of already-computed frames, audio
    segments, and other things."""
    def __init__(self, directory, frame_format='png'):
        self.directory = directory
        self.cache = None
        self.frame_format = frame_format

        assert self.directory[0] == '/', \
          'Creating cache with a relative path.  This can cause problems later if the ' \
          'current directory changes, which is not unlikely to happen.'

    def scan_directory(self):
        """Examine the cache directory and remember what we see there."""
        self.cache = {}
        try:
            for cached_frame in os.listdir(self.directory):
                self.cache[os.path.join(self.directory, cached_frame)] = True
        except FileNotFoundError:
            os.makedirs(self.directory)

        counts = '; '.join(map(lambda x: f'{x[1]} {x[0]}',
          collections.Counter(map(lambda x: os.path.splitext(x)[1][1:],
          self.cache.keys())).items()))
        print(f'Found {len(self.cache)} cached items ({counts}) in {self.directory}')

    def clear(self):
        """ Delete all the files in the cache. """
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        self.cache = None

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

    def request_frame(self, t):
        """Called during the rendering process, before any get_frame calls, to
        indicate that a frame at the given t will be needed in the future.  Can
        help if frames are generated in batches, such as in from_file.  Default
        is to do nothing. """

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
        """ Fully realize this clip, ensuring that no exceptions occur and
        that the right sizes of video frames and audio samples are returned.
        Useful for testing. """

        self.metrics.verify()

        require_positive(frame_rate, 'frame rate')

        for t in frame_times(self.length(), frame_rate):
            self.request_frame(t)

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

    def stage(self, directory, cache, frame_rate, fname=""):
        """ Get everything for this clip onto to disk in the specified
        directory:  Symlinks to each frame and a flac file of the audio. """

        # Do things in the requested directory.
        with temporarily_changed_directory(directory):
            # Audio.
            audio_fname = 'audio.flac'
            data = self.get_samples()
            assert data is not None
            soundfile.write(audio_fname, data, self.sample_rate())

            # Video.
            task = f"Staging {fname}" if fname else "Staging"

            fts = list(frame_times(self.length(), frame_rate))
            for t in fts:
                self.request_frame(t)

            with custom_progressbar(task, len(fts)) as pb:
                for index, t in enumerate(fts):
                    # Make sure this frame is in the cache, and figure out
                    # where.
                    cached_filename = self.get_cached_filename(cache, t)

                    # Add a symlink from this frame in the cache to the
                    # staging area.
                    os.symlink(cached_filename, f'{index:06d}.{cache.frame_format}')

                    # Update the progress bar.
                    pb.update(index)


    def save(self, fname, frame_rate, bitrate=None, target_size=None, two_pass=False,
             preset='slow', cache_dir='/tmp/clipcache/computed_frames'):
        """ Save to a file.

        Bitrate controls the target bitrate.  Handles the tradeoff between
        file size and output quality.

        Target size specifies the file size we want, in MB.

        At most one of bitrate and target_size should be given.  If both are
        omitted, the default is to target a bitrate of 1024k.

        Preset controls how quickly ffmpeg encodes.  Handles the tradeoff
        between encoding speed and output quality.  Choose from:
            ultrafast superfast veryfast faster fast medium slow slower veryslow
        The documentation for these says to "use the slowest preset you have
        patience for."
        """

        # First, a simple case: If we're saving to an audio-only format, it's
        # easy.
        if re.search('.(flac|wav)$', fname):
            data = self.get_samples()
            assert data is not None
            soundfile.write(fname, data, self.sample_rate())
            return

        # So we need to save video.  First, construct the complete path name.
        # We'll need this during the ffmpeg step, because that runs in a
        # temporary current directory.
        full_fname = os.path.join(os.getcwd(), fname)

        # Force the frame cache to be read before we change to the temp
        # directory.
        cache = ClipCache(cache_dir)
        cache.scan_directory()

        # Figure out what bitrate to target.
        if bitrate is None and target_size is None:
            # A hopefully sensible high-quality default.
            bitrate = '1024k'
        elif bitrate is None and target_size is not None:
            # Compute target bit rate, which should be in bits per second,
            # from the target filesize.
            target_bytes = 2**20 * target_size
            target_bits = 8*target_bytes
            bitrate = target_bits / self.length()
            bitrate -= 128*1024 # Audio defaults to 1024 kilobits/second.
        elif bitrate is not None and target_size is None:
            # Nothing to do -- just use the bitrate as given.
            pass
        else:
            raise ValueError("Specify either bitrate or target_size, not both.")

        with tempfile.TemporaryDirectory() as td:
            # Fill the temporary directory with the audio and a bunch of
            # (symlinks to) individual frames.
            self.stage(directory=td,
                       cache=cache,
                       frame_rate=frame_rate,
                       fname=fname)

            # Invoke ffmpeg to assemble all of these into the completed video.
            with temporarily_changed_directory(td):
                # Some shared arguments across all ffmpeg calls: single pass,
                # first pass of two, and second pass of two.
                # These include filters to:
                # - Ensure that the width and height are even, padding with a
                #   black row or column if needed.
                # - Set the pixel format to yuv420p, which seems to be needed
                #   to get outputs that play on Apple gadgets.
                # - Set the output frame rate.
                args = [
                    f'-framerate {frame_rate}',
                    f'-i %06d.{cache.frame_format}',
                    '-i audio.flac',
                    '-vcodec libx264',
                    '-f mp4',
                    f'-vb {bitrate}' if bitrate else '',
                    f'-preset {preset}' if preset else '',
                    '-profile:v high',
                    f'-filter_complex "format=yuv420p,pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2,fps={frame_rate}"', #pylint: disable=line-too-long
                ]

                num_frames = int(self.length() * frame_rate)

                if not two_pass:
                    ffmpeg(task=f"Encoding {fname}",
                           *(args + [f'{full_fname}']),
                           num_frames=num_frames)
                else:
                    ffmpeg(task=f"Encoding {fname}, pass 1",
                           *(args + ['-pass 1', '/dev/null']),
                           num_frames=num_frames)
                    ffmpeg(task=f"Encoding {fname}, pass 2",
                           *(args + ['-pass 2', f'{full_fname}']),
                           num_frames=num_frames)

            print(f'Wrote {self.readable_length()} to {fname}.')

    def get_cached_filename(self, cache, t):
        """Make sure the frame is in the cache given, computing it if
        necessary, and return its filename."""
        # Look for the frame we need in the cache.
        cached_filename, success = cache.lookup(self.frame_signature(t),
                                                cache.frame_format)

        # If it wasn't there, generate it and save it there.
        if not success:
            self.compute_and_cache_frame(t, cache, cached_filename)

        # Done!
        return cached_filename

    def compute_and_cache_frame(self, t, cache, cached_filename):
        """Call get_frame to compute one frame, and put it in the cache."""
        # Get the frame.
        frame = self.get_frame(t)

        assert frame is not None, ("A clip of type " + str(type(self)) +
          " returned None instead of a real frame.")

        # Make sure we got a legit frame.
        assert frame is not None, \
          "Got None instead of a real frame for " + str(self.frame_signature(t))
        assert frame.shape[1] == self.width(), f"For {self.frame_signature(t)}," \
            f"I got a frame of width {frame.shape[1]} instead of {self.width()}."
        assert frame.shape[0] == self.height(), f"For {self.frame_signature(t)}," \
            f"I got a frame of height {frame.shape[0]} instead of {self.height()}."

        # Add to disk and to the cache.
        cv2.imwrite(cached_filename, frame)
        cache.insert(cached_filename)

        # Done!
        return frame

    def get_frame_cached(self, cache, t):
        """Return a frame, from the cache if possible, computed from scratch
        if needed."""
        # Look for the frame we need in the cache.
        cached_filename, success = cache.lookup(self.frame_signature(t), cache.frame_format)

        # Did we find it?
        if success:
            # Yes.  Read from disk.
            return read_image(cached_filename)
        else:
            # No. Generate and save to disk for next time.
            return self.compute_and_cache_frame(t, cache, cached_filename)



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


def get_duration_from_ffprobe_stream(stream):
    """ Given a dictionary of ffprobe-returned attributes of a stream, try to
    figure out the duration of that stream. """

    if 'duration' in stream and is_float(stream['duration']):
        return float(stream['duration'])

    if 'tag:DURATION' in stream:
        if match := re.match(r"(\d\d):(\d\d):([0-9\.]+)", stream['tag:DURATION']):
            hours = float(match.group(1))
            mins = float(match.group(2))
            secs = float(match.group(3))
            return secs + 60*mins + 60*60*hours

    raise ValueError(f"Could not find a duration in ffprobe stream. {stream}")

def metrics_and_frame_rate_from_stream_dicts(video_stream, audio_stream, fname):
    """ Given dicts representing the audio and video elements of a clip,
    return the appropriate metrics object. """
    # Some videos, especially from mobile phones, contain metadata asking for a
    # rotation.  We'll generally not try to deal with that here ---better,
    # perhaps, to let the user flip or rotate as needed later--- but
    # landscape/portait differences are important because they affect the width
    # and height, and are respected by ffmpeg when the frames are extracted.
    # Thus, we need to apply that change here to prevent mismatched frame sizes
    # down the line.
    if (video_stream and 'tag:rotate' in video_stream
          and video_stream['tag:rotate'] in ['-90','90']):
        video_stream['width'],video_stream['height'] = video_stream['height'],video_stream['width']

    if video_stream and audio_stream:
        vlen = get_duration_from_ffprobe_stream(video_stream)
        alen = get_duration_from_ffprobe_stream(audio_stream)
        if abs(vlen - alen) > 0.5:
            raise ValueError(f"In {fname}, video length ({vlen}) and audio length ({alen}) "
              "do not match. Perhaps load video and audio separately?")

        return Metrics(width = eval(video_stream['width']),
                       height = eval(video_stream['height']),
                       sample_rate = eval(audio_stream['sample_rate']),
                       num_channels = eval(audio_stream['channels']),
                       length = min(vlen, alen)), \
               eval(video_stream['avg_frame_rate']), \
               True, True
    elif video_stream:
        vlen = get_duration_from_ffprobe_stream(video_stream)
        return Metrics(src = Clip.default_metrics,
                       width = eval(video_stream['width']),
                       height = eval(video_stream['height']),
                       length = vlen), \
               eval(video_stream['avg_frame_rate']), \
               True, False
    elif audio_stream:
        alen = get_duration_from_ffprobe_stream(audio_stream)
        return Metrics(src = Clip.default_metrics,
                       sample_rate = eval(audio_stream['sample_rate']),
                       num_channels = eval(audio_stream['channels']),
                       length = alen), None, False, True
    else:
        # Should be impossible to get here, but just in case...
        raise ValueError(f"File {fname} contains neither audio nor video.") # pragma: no cover

def metrics_from_ffprobe_output(ffprobe_output, fname, suppress_video=False, suppress_audio=False):
    """ Given the output of a run of ffprobe -of compact -show_entries
    stream, return a Metrics object based on that data, or complain if
    something strange is in there. """

    video_stream = None
    audio_stream = None

    for line in ffprobe_output.strip().split('\n'):
        stream = {}
        fields = line.split('|')
        if fields[0] != 'stream': continue
        for pair in fields[1:]:
            key, val = pair.split('=')
            assert key not in stream
            stream[key] = val

        if stream['codec_type'] == 'video' and not suppress_video:
            if video_stream is not None:
                raise ValueError(f"Don't know what to do with {fname},"
                  "which has multiple video streams.")
            video_stream = stream
        elif stream['codec_type'] == 'video' and suppress_video:
            pass
        elif stream['codec_type'] == 'audio' and not suppress_audio:
            if audio_stream is not None:
                raise ValueError(f"Don't know what to do with {fname},"
                  "which has multiple audio streams.")
            audio_stream = stream
        elif stream['codec_type'] == 'audio' and suppress_audio:
            pass
        elif stream['codec_type'] == 'data':
            pass
        else:
            raise ValueError(f"Don't know what to do with {fname}, "
              f"which has an unknown stream of type {stream['codec_type']}.")

    return metrics_and_frame_rate_from_stream_dicts(video_stream, audio_stream, fname)

def audio_samples_from_file(fname, cache, expected_sample_rate, expected_num_channels,
  expected_num_samples):
    """ Extract audio data from a file, which may be either a pure audio
    format or a video file containing an audio stream."""

    # Grab the file's extension.
    ext = os.path.splitext(fname)[1].lower()

    # Is it in a format we know how to read directly?  If so, read it and
    # declare victory.
    direct_formats = list(map(lambda x: "." + x.lower(),
      soundfile.available_formats().keys()))
    if ext in direct_formats:
        print("Reading audio from", fname)

        # Acquire the data from the file.
        data, sample_rate = soundfile.read(fname, always_2d=True)

        # Complain if the sample rates or numbers of channels don't match.
        if sample_rate != expected_sample_rate:
            raise ValueError(f"From {fname}, expected sample rate {expected_sample_rate},"
                f" but found {sample_rate} instead.")
        if data.shape[1] != expected_num_channels:
            raise ValueError(f"From {fname}, expected {expected_num_channels} channels,"
                f" but found {data.shape[1]} instead.")

        # Complain if there's a length mismatch longer than about half a
        # second.
        if abs(data.shape[0] - expected_num_samples) > expected_sample_rate:
            raise ValueError(f"From {fname}, got {data.shape[0]}"
              f" samples instead of {expected_num_samples}.")

        # If there's a small mismatch, just patch it.
        # - Too short?
        if data.shape[0] < expected_num_samples:
            new_data = np.zeros([expected_num_samples, expected_num_channels], dtype=np.float64)
            new_data[0:data.shape[0],:] = data
            data = new_data
        # - Too long?
        if data.shape[0] > expected_num_samples:
            data = data[:expected_num_samples,:]

        return data

    # Not a format we can read directly.  Instead, let's use ffmpeg to get it
    # indirectly.  (...or simply pull it from the cache, if it happens to be
    # there.)
    cached_filename, success = cache.lookup(fname, 'flac')
    if not success:
        print(f'Extracting audio from {fname}')
        assert '.flac' in direct_formats
        full_fname = os.path.join(os.getcwd(), fname)
        with temporary_current_directory():
            audio_fname = 'audio.flac'
            ffmpeg(
                f'-i {full_fname}',
                '-vn',
                f'{audio_fname}',
            )
            os.rename(audio_fname, cached_filename)
            cache.insert(cached_filename)

    return audio_samples_from_file(cached_filename,
                                   cache,
                                   expected_sample_rate,
                                   expected_num_channels,
                                   expected_num_samples)

class from_file(Clip):
    """ A clip read from a file such as an mp4, flac, or other format readable
    by ffmpeg. """

    def __init__(self, fname, suppress_video=False, suppress_audio=False):
        """ Video decoding happens in batches.  Use decode_chunk_length to
        specify the number of seconds of frames decoded in each batch.
        Larger values reduce the overhead of starting the decode process,
        but may waste time decoding frames that are never used.  Use None to
        decode the entire video at once. """
        super().__init__()

        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Could not open file {fname}.")
        self.fname = os.path.abspath(fname)

        cache_dir = os.path.join('/tmp/clipcache', re.sub('/', '_', self.fname))
        self.cache = ClipCache(cache_dir)

        self.acquire_metrics(suppress_video, suppress_audio)
        self.samples = None

        self.exploded = False

    def acquire_metrics(self, suppress_video, suppress_audio):
        """ Set the metrics attribute, either by grabbing the metrics from the
        cache, or by getting them the hard way via ffprobe."""

        # Do we have the metrics in the cache?
        digest = hashlib.md5(self.fname.encode()).hexdigest()
        cached_filename, exists = self.cache.lookup(digest, 'dim')

        if exists:
            # Yes. Grab it.
            print(f"Using cached dimensions for {self.fname}")
            with open(cached_filename, 'r') as f:
                deets = f.read()
        else:
            # No.  Get the metrics, then store in the cache for next time.
            print(f"Probing dimensions for {self.fname}")
            with subprocess.Popen(f'ffprobe -hide_banner -v error {self.fname} '
                                  '-of compact -show_entries stream', shell=True,
                                  stdout=subprocess.PIPE) as proc:
                deets = proc.stdout.read().decode('utf-8')

            with open(cached_filename, 'w') as f:
                print(deets, file=f)
            self.cache.insert(cached_filename)

        # Parse the (very detailed) ffprobe response to get the metrics we
        # need.
        response = metrics_from_ffprobe_output(deets, self.fname, suppress_video, suppress_audio)
        self.metrics, self.frame_rate, self.has_video, self.has_audio = response

    def frame_signature(self, t):
        require_positive(t, "timestamp")

        index = int(t * self.frame_rate)

        if self.has_video:
            return [self.fname, index]
        else:
            return ['solid', {
                'width': self.metrics.width,
                'height': self.metrics.height,
                'color': [0,0,0,255]
            }]

    def request_frame(self, t):
        require_positive(t, "timestamp")
        self.exploded = False

    def get_frame(self, t):
        require_positive(t, "timestamp")

        if self.has_video:
            # Make sure we've extracted all of the frames we'll need into the
            # cache.
            if not self.exploded:
                self.explode()

            # Return the frame from the cache.
            frame = self.get_frame_cached(self.cache, t)
            assert frame is not None
            return frame
        else:
            return np.zeros([self.metrics.height, self.metrics.width, 4], np.uint8)

    def explode(self):
        """Expand the requested frames into our cache for later."""
        assert self.has_video

        with temporary_current_directory():
            start_index = 0
            num_frames = int(self.length() * self.frame_rate)
            end_index = start_index + num_frames

            # Extract the frames into the current temporary directory.
            ffmpeg(
                f'-ss {start_index/self.frame_rate}',
                f'-t {self.length()}',
                f'-i {self.fname}',
                f'-r {self.frame_rate}',
                f'%06d.{self.cache.frame_format}',
                task=f'Exploding {os.path.basename(self.fname)}',
                num_frames=num_frames
            )

            # Add each frame to the cache.
            rng = range(start_index, end_index)
            for index in rng:
                seq_fname = os.path.abspath(f'{index-start_index+1:06d}.{self.cache.frame_format}')
                t = (index+0.5)/self.frame_rate
                cached_fname, exists = self.cache.lookup(self.frame_signature(t),
                                                         self.cache.frame_format)
                if not exists:
                    try:
                        os.rename(seq_fname, cached_fname)
                    except FileNotFoundError: #pragma: no cover
                        # If we get here, that means ffmpeg thought a frame
                        # should exist, that ultimately was not extracted.
                        # This seems to happen from mis-estimations of the
                        # video length, or sometimes from simply missing
                        # frames.  To keep things rolling, let's fill in a
                        # black frame instead.
                        print(f"[Exploding {self.fname} did not produce frame index={index},t={t}. "
                          "Using black instead.]")
                        fr = np.zeros([self.height(), self.width(), 3], np.uint8)
                        cv2.imwrite(cached_fname, fr)

                    self.cache.insert(cached_fname)

        self.exploded = True

    def get_samples(self):
        if self.samples is None:
            if self.has_audio:
                self.samples = audio_samples_from_file(self.fname,
                                                       self.cache,
                                                       self.sample_rate(),
                                                       self.num_channels(),
                                                       self.num_samples())
            else:
                self.samples = np.zeros([self.metrics.num_samples(), self.metrics.num_channels])
        return self.samples



