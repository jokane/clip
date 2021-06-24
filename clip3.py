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
video_file), by starting from a blank video (see: solid), or by using
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

# pylint: disable=too-many-lines

from abc import ABC, abstractmethod
import contextlib
import collections
from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import os
from pprint import pprint # pylint: disable=unused-import
import re
import shutil
import subprocess
import tempfile
import time
import threading
import typing

import cv2
import numpy as np
import progressbar
from PIL import Image, ImageFont, ImageDraw
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
        raise exception_class(f'Expected {name} to be a {condition}, but got {x} instead.')

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

def require_positive(x, name):
    """ Raise an informative exception if x is not positive. """
    require(x, is_positive, "positive", name, ValueError)

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
        raise ValueError(f'Expected {name1} less than or equak to {name2},'
          f'but it is not. {x} > {y}')

def require_less(x, y, name1, name2):
    """ Raise an informative exception if x is greater than y. """
    if x >= y:
        raise ValueError(f'Expected {name1} less than {name2},'
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

def get_font(font, size):
    """
    Return a TrueType font for use on Pillow images, with caching to prevent
    loading the same font again and again.    (The performance improvement seems to
    be small but non-zero.)
    """
    if (font, size) not in get_font.cache:
        try:
            get_font.cache[(font, size)] = ImageFont.truetype(font, size)
        except OSError as e:
            raise ValueError(f"Failed to open font {font}.") from e
    return get_font.cache[(font, size)]
get_font.cache = dict()


@dataclass
class Metrics:
    """ A object describing the dimensions of a Clip. """
    width: int
    height: int
    frame_rate: float
    sample_rate: int
    num_channels: int
    length: float

    def __init__(self, src=None, width=None, height=None, frame_rate=None,
                 sample_rate=None, num_channels=None, length=None):
        assert src is None or isinstance(src, Metrics), f'src should be Metrics, not {type(src)}'
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

    def verify_compatible_with(self, other, check_video=True, check_audio=True, check_length=False):
        """ Make sure two Metrics objects match each other.  Complain if not. """
        assert isinstance(other, Metrics)

        if check_video:
            require_equal(self.width, other.width, "widths")
            require_equal(self.height, other.height, "heights")
            require_equal(self.frame_rate, other.frame_rate, "frame rates")

        if check_audio:
            require_equal(self.num_channels, other.num_channels, "numbers of channels")
            require_equal(self.sample_rate, other.sample_rate, "sample rates")

        if check_length:
            require_equal(self.length, other.length, "lengths")

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

    def clear(self):
        """ Delete all the files in the cache. """
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
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

frame_cache_format = 'png'

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

    def width(self):
        """Width of the video, in pixels."""
        return self.metrics.width

    def height(self):
        """Height of the video, in pixels."""
        return self.metrics.height

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
        mins = int(mins)
        hours = int(hours)
        secs = int(secs)
        if hours > 0:
            return f'{hours}:{mins:02}:{secs:02}'
        else:
            return f'{mins}:{secs:02}'

    def compute_and_cache_frame(self, index, cached_filename):
        """Call get_frame to compute one frame, and put it in the cache."""
        # Get the frame.
        frame = self.get_frame(index)
        assert frame is not None, ("A clip of type " + type(self) +
          " returned None instead of a real frame.")

        # Make sure we got a legit frame.
        assert frame is not None, \
          "Got None instead of a real frame for " + str(self.frame_signature(index))
        assert frame.shape[1] == self.width(), \
          "For %s, I got a frame of width %d instead of %d." % \
          (self.frame_signature(index), frame.shape[1], self.width())
        assert frame.shape[0] == self.height(), \
          "From %s, I got a frame of height %d instead of %d." %  \
          (self.frame_signature(index), frame.shape[0], self.height())

        # Add to disk and to the cache.
        cv2.imwrite(cached_filename, frame)
        cache.insert(cached_filename)

        # Done!;
        return frame

    def get_frame_cached(self, index):
        """Return a frame, from the cache if possible, computed from scratch
        if needed."""

        # Look for the frame we need in the cache.
        cached_filename, success = cache.lookup(
          self.frame_signature(index),
          frame_cache_format)

        # Did we find it?
        if success:
            # Yes.  Read from disk.
            frame = cv2.imread(cached_filename, cv2.IMREAD_UNCHANGED)
            assert frame is not None
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
                assert frame is not None
            assert frame.shape == (self.height(), self.width(), 4)
            return frame
        else:
            # No. Generate and save to disk for next time.
            return self.compute_and_cache_frame(index, cached_filename)

    def get_cached_filename(self, index):
        """Make sure the frame is in the cache, computing it if necessary,
        and return its filename."""
        # Look for the frame we need in the cache.
        cached_filename, success = cache.lookup(
          self.frame_signature(index),
          frame_cache_format
        )

        # If it wasn't there, generate it and save it there.
        if not success:
            self.compute_and_cache_frame(index, cached_filename)

        # Done!
        return cached_filename

    def save(self, fname, bitrate='1024k', preset='slow'):
        """Save to a file."""

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
        if cache.cache is None:
            cache.scan_directory()

        with temporary_current_directory():
            audio_fname = 'audio.flac'
            data = self.get_samples()
            assert data is not None
            soundfile.write(audio_fname, data, self.sample_rate())

            with custom_progressbar(f"Staging {fname}", self.num_frames()) as pb:
                for index in range(0, self.num_frames()):

                    # Make sure this frame is in the cache, and figure out
                    # where.
                    cached_filename = self.get_cached_filename(index)

                    # Add a symlink from this frame in the cache to the
                    # staging area.
                    os.symlink(cached_filename, f'{index:06d}.{frame_cache_format}')

                    # Update the progress bar.
                    pb.update(index)

            # We have a directory containing the audio and a bunch of
            # (symlinks to) individual frames.   Invoke ffmpeg to assemble
            # all of these into the completed video.
            ffmpeg(
                f'-framerate {self.frame_rate()}',
                f'-i %06d.{frame_cache_format}',
                f'-i {audio_fname}',
                '-vcodec libx264',
                '-f mp4 ',
                f'-vb {bitrate}' if bitrate else '',
                f'-preset {preset}' if preset else '',
                '-profile:v high',
                '-filter_complex "color=black,format=rgb24[c];[c][0]scale2ref[c][i];[c][i]overlay=format=auto:shortest=1,setsar=1,format=yuv420p"', #pylint: disable=line-too-long
                f'{full_fname}',
                task=f"Encoding {fname}",
                num_frames=self.num_frames()
            )

            print(f'Wrote {self.readable_length()} to {fname}.')

    def preview(self):
        """Render the video part and display it in a window on screen."""
        with custom_progressbar("Previewing", self.length()) as pb:
            for i in range(0, self.length()):
                frame = self.get_frame_cached(i)
                pb.update(i)
                cv2.imshow("", frame)
                cv2.waitKey(1)

        cv2.destroyWindow("")

    def verify(self):
        """ Fully realize a clip, ensuring that no exceptions occur and that
        the right sizes of video frames and audio samples are returned. """
        with custom_progressbar(task="Verifying", steps=self.num_frames()) as pb:
            for i in range(self.num_frames()):
                self.frame_signature(i)
                frame = self.get_frame(i)
                assert isinstance(frame, np.ndarray), f'{type(frame)} ({frame})'
                assert frame.shape == (self.height(), self.width(), 4)
                pb.update(i)

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
    """ Inherit from this for Clip classes that only really have audio, to default to
    simple black frames for the video. """
    def __init__(self):
        super().__init__()
        self.color = [0, 0, 0, 255]
        self.frame = None

    def frame_signature(self, index):
        return ['solid', {
            'width': self.metrics.width,
            'height': self.metrics.height,
            'color': self.color
        }]

    def get_frame(self, index):
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

    def frame_signature(self, index):
        return self.clip.frame_signature(index)

    def get_frame(self, index):
        return self.clip.get_frame(index)

    def get_samples(self):
        return self.clip.get_samples()

class solid(Clip):
    """A video clip in which each frame has the same solid color."""
    def __init__(self, color, width, height, frame_rate, length):
        super().__init__()
        assert is_color(color)
        self.metrics = Metrics(
            default_metrics,
            width=width,
            height=height,
            frame_rate=frame_rate,
            length=length
        )

        self.color = [color[2], color[1], color[0], 255]
        self.frame = None

    # Avoiding both code duplication and multiple inheritance here...
    frame_signature = AudioClip.frame_signature
    get_frame = AudioClip.get_frame
    get_samples = VideoClip.get_samples

def black(width, height, frame_rate, length):
    """ A silent solid black clip. """
    return solid([0,0,0], width, height, frame_rate, length)

def white(width, height, frame_rate, length):
    """ A silent white black clip. """
    return solid([255,255,255], width, height, frame_rate, length)

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
        self.metrics = Metrics(
          default_metrics,
          length = length,
          sample_rate = sample_rate,
          num_channels = num_channels
        )

    def get_samples(self):
        samples = np.arange(self.num_samples()) / self.sample_rate()
        samples = self.volume * np.sin(2 * np.pi * self.frequency * samples)
        samples = np.stack([samples]*self.num_channels(), axis=1)
        return samples

class join(Clip):
    """ Create a new clip that combines the video of one clip with the audio of
    another. """
    def __init__(self, video_clip, audio_clip):
        super().__init__()

        require_clip(video_clip, "video clip")
        require_clip(audio_clip, "audio clip")
        require_equal(video_clip.length(), audio_clip.length(), "clip lengths")
        assert not isinstance(video_clip, AudioClip)
        assert not isinstance(audio_clip, VideoClip)
        assert not isinstance(audio_clip, solid)

        self.video_clip = video_clip
        self.audio_clip = audio_clip

        self.metrics = Metrics(
          src=video_clip.metrics,
          sample_rate=audio_clip.sample_rate(),
          num_channels=audio_clip.num_channels()
        )

    def frame_signature(self, index):
        return self.video_clip.frame_signature(index)
    def get_frame(self, index):
        return self.video_clip.get_frame(index)
    def get_samples(self):
        return self.audio_clip.get_samples()

@dataclass
class Element:
    """An element to be included in a composite."""
    class VideoMode(Enum):
        """ How should the video for this element be composited into the
        final clip?"""
        REPLACE = 1
        BLEND = 2

    class AudioMode(Enum):
        """ How should the video for this element be composited into the
        final clip?"""
        REPLACE = 1
        ADD = 2

    clip : Clip
    start_time : float
    position : typing.Tuple[int, int]
    video_mode : VideoMode = VideoMode.REPLACE
    audio_mode : AudioMode = AudioMode.REPLACE

    def apply_to_frame(self, under, index, make_frame):
        """ Compute the frame signature that results from applying this
        element (if make_frame==False), or actually compute the frame (if
        make_frame==False).  The under parameter should be existing signature
        or the existing frame, respectively for those two cases."""

        # If this element does not apply at this time, make no change.
        start_index = int(self.start_time*self.clip.frame_rate())
        if index < start_index or index >= start_index + self.clip.num_frames():
            return under

        # If there's no frame here yet, or if we're a replace operation, ignore
        # the existing thing and just use ours.
        if make_frame and (under is None or self.video_mode == Element.VideoMode.REPLACE):
            return self.clip.get_frame(index-start_index)
        if not make_frame and (under == "" or self.video_mode == Element.VideoMode.REPLACE):
            return self.clip.frame_signature(index-start_index)

        # Should we alpha-blend ourselves in?
        if self.video_mode == Element.VideoMode.BLEND:
            if make_frame:
                over_frame = self.clip.get_frame(index-start_index)
                over_bgr = over_frame[:,:,:3]
                over_alpha = over_frame[:,:,3]/255
                over_alpha = np.stack([over_alpha]*3, axis=2)
                under[:,:,:3] = over_alpha*over_bgr + (1-over_alpha)*under[:,:,:3]
                return under
            else:
                sig = self.clip.frame_signature(index-start_index)
                return ['blend', sig, under]

        # Should never get here.
        assert False #pragma: no cover

class composite(Clip):
    """ Given a collection of elements, form a composite clip."""
    def __init__(self, *args):
        super().__init__()

        self.elements = list(args)

        # Sanity check on the inputs.
        for (i, e) in enumerate(self.elements):
            assert isinstance(e, Element)
            require_clip(e.clip, f'clip {i}')
            require_float(e.start_time, f'start time {i}')
            require_non_negative(e.start_time, f'start time {i}')

        # Compute our metrics.  Same as all of the clips, except for the
        # length computed from ending time of the final-ending clip.
        length = 0
        for (i, e) in enumerate(self.elements):
            length = max(length, e.start_time + e.clip.length())
        self.metrics = Metrics(
          src=self.elements[0].clip.metrics,
          length=length
        )

        # Check for metric mismatches.
        for (i, e) in enumerate(self.elements[1:]):
            self.metrics.verify_compatible_with(e.clip.metrics)

    def frame_signature(self, index):
        sig = ""
        for e in self.elements:
            sig = e.apply_to_frame(sig, index, False)
        if sig == "":
            sig = ['solid', {
              'width': self.metrics.width,
              'height': self.metrics.height,
              'color': [0, 0, 0, 0]
            }]
        return sig

    def get_frame(self, index):
        frame = None
        for e in self.elements:
            frame = e.apply_to_frame(frame, index, True)
        if frame is None:
            frame = np.zeros([self.metrics.height, self.metrics.width, 4], np.uint8)
        return frame

    def get_samples(self):
        samples = np.zeros([self.metrics.num_samples(), self.metrics.num_channels])
        for e in self.elements:
            clip_samples = e.clip.get_samples()
            start_sample = int(e.start_time*e.clip.sample_rate())
            end_sample = start_sample + e.clip.num_samples()
            if e.audio_mode == Element.AudioMode.REPLACE:
                samples[start_sample:end_sample] = clip_samples
            elif e.audio_mode == Element.AudioMode.ADD:
                samples[start_sample:end_sample] += clip_samples

        return samples

def chain(*args):
    """ Concatenate a series of clips.  The clips may be given individually,
    in lists or other iterables, or a mixture of both.  """
    return fade_chain(0, *args)

def fade_chain(fade, *args):
    """ Concatenate a series of clips, with a given amount of overlap between
    each successive pair.  The clips may be given individually, in lists or
    other iterables, or a mixture of both.  """

    # Construct our list of clips.  Flatten each list; keep each individual
    # clip.
    clips = list()
    for x in args:
        if is_iterable(x):
            clips += x
        else:
            clips.append(x)

    # Sanity checks.
    require_float(fade, "fade time")
    require_non_negative(fade, "fade time")

    for clip in clips:
        require_clip(clip, "clip")

    if len(clips) == 0:
        raise ValueError("Need at least one clip to form a chain.")

    # Figure out when each clip should start and make a list of elements for
    # composite.
    start_time = 0
    elements = list()
    for i, clip in enumerate(clips):
        if i>0 and fade>0:
            clip = scale_alpha(clip, lambda index: min(index/clip.frame_rate()/fade, 1.0))
        if i<len(clips)-1 and fade>0:
            clip = scale_alpha(clip,
              lambda index: min((clip.num_frames()-index)/clip.frame_rate()/fade, 1.0))

        elements.append(Element(
          clip=clip,
          start_time=start_time,
          position=(0,0),
          video_mode=Element.VideoMode.BLEND,
          audio_mode=Element.AudioMode.ADD,
        ))
        start_time += clip.length() - fade

    # Let composite do all the work.
    return composite(*elements)

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
        self.metrics = Metrics(clip.metrics)

    def frame_signature(self, index):
        factor = self.factor(index)
        require_float(factor, f'factor at index {index}')
        return ['scale_alpha', self.clip.frame_signature(index), factor]

    def get_frame(self, index):
        factor = self.factor(index)
        require_float(factor, f'factor at index {index}')
        frame = self.clip.get_frame(index)
        if factor != 1.0:
            frame = frame.astype('float')
            frame[:,:,3] *= factor
            frame = frame.astype('uint8')
        return frame

def metrics_from_ffprobe_output(ffprobe_output, fname):
    """ Given the output of a run of ffprobe -of compact -show_entries
    stream, return a Metrics object based on that data, or complain if
    something strange is in there. """

    video_stream = None
    audio_stream = None

    for line in ffprobe_output.strip().split('\n'):
        stream = dict()
        for pair in line[7:].split('|'):
            key, val = pair.split('=')
            assert key not in stream
            stream[key] = val

        if stream['codec_type'] == 'video':
            if video_stream is not None:
                raise ValueError(f"Don't know what to do with {fname},"
                  "which has multiple video streams.")
            video_stream = stream
        elif stream['codec_type'] == 'audio':
            if audio_stream is not None:
                raise ValueError(f"Don't know what to do with {fname},"
                  "which has multiple audio streams.")
            audio_stream = stream
        else:
            raise ValueError(f"Don't know what to do with {fname},"
              "which has an unknown stream of type {stream['codec_type']}.")

    if video_stream and audio_stream:
        vlen = float(video_stream['duration'])
        alen = float(audio_stream['duration'])
        if abs(vlen - alen) > 0.5:
            raise ValueError(f"In {fname}, video length ({vlen}) and audio length ({alen}) "
              "do not match. Perhaps load video and audio separately?")

        return Metrics(
          width = eval(video_stream['width']),
          height = eval(video_stream['height']),
          frame_rate = eval(video_stream['avg_frame_rate']),
          sample_rate = eval(audio_stream['sample_rate']),
          num_channels = eval(audio_stream['channels']),
          length = min(vlen, alen)
        ), True, True
    elif video_stream:
        return Metrics(
          src = default_metrics,
          width = eval(video_stream['width']),
          height = eval(video_stream['height']),
          frame_rate = eval(video_stream['avg_frame_rate']),
          length = eval(video_stream['duration'])
        ), True, False
    elif audio_stream:
        return Metrics(
          src = default_metrics,
          sample_rate = eval(audio_stream['sample_rate']),
          num_channels = eval(audio_stream['channels']),
          length = eval(audio_stream['duration'])
        ), False, True
    else:
        # Should be impossible to get here, but just in case...
        raise ValueError(f"File {fname} contains neither audio nor video.") # pragma: no cover

def audio_samples_from_file(fname, expected_sample_rate, expected_num_channels,
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
            new_data = np.zeros([expected_num_samples, expected_num_channels], dtype=np.uint8)
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
    return audio_samples_from_file(
      cached_filename,
      expected_sample_rate,
      expected_num_channels,
      expected_num_samples
    )


class from_file(Clip):
    """ Create a clip from a file such as an mp4, flac, or other format
    readable by ffmpeg. """
    def __init__(self, fname, decode_chunk_length=10, forced_length=None):
        """ Video decoding happens in batches.  Use decode_chunk_length to
        specify the number of seconds of frames decoded in each batch.
        Larger values reduce the overhead of starting the decode process,
        but may waste time decoding frames that are never used.  Use None to
        decode the entire video at once. """
        super().__init__()

        # Make sure the file exists.
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Could not open file {fname}.")
        self.fname = os.path.abspath(fname)

        self.acquire_metrics(forced_length=forced_length)
        self.decode_chunk_length = decode_chunk_length
        self.samples = None

    def acquire_metrics(self, forced_length=None):
        """ Set the metrics attribute, either by grabbing the metrics from the
        cache, or by getting them the hard way via ffprobe."""

        # Do we have the metrics in the cache?
        cached_filename, exists = cache.lookup(hashlib.md5(self.fname.encode()).hexdigest(), 'dim')
        if exists:
            # Yes. Grab it.
            print(f"Using cached dimensions for {self.fname}")
            with open(cached_filename, 'r') as f:
                deets = f.read()
        else:
            # No.  Get the metrics, then store in the cache for next time.
            print(f"Probing dimensions for {self.fname}")
            with subprocess.Popen(f'ffprobe -hide_banner -count_frames -v error {self.fname} '
                  '-of compact -show_entries stream', shell=True, stdout=subprocess.PIPE) as proc:
                deets = proc.stdout.read().decode('utf-8')
            with open(cached_filename, 'w') as f:
                print(deets, file=f)
            cache.insert(cached_filename)

        # Parse the (very detailed) ffprobe response to get the metrics we
        # need.
        response = metrics_from_ffprobe_output(deets, self.fname)
        self.metrics, self.has_video, self.has_audio = response

        # Adjust the length if our user insists.
        if forced_length is not None:
            self.metrics = Metrics(self.metrics, length=forced_length)

    def frame_signature(self, index):
        require_int(index, "frame index")
        require_non_negative(index, "frame index")
        if self.has_video:
            return [self.fname, index]
        else:
            return ['solid', {
                'width': self.metrics.width,
                'height': self.metrics.height,
                'color': [0,0,0,255]
            }]

    def get_frame(self, index):
        require_int(index, "frame index")
        require_non_negative(index, "frame index")

        if self.has_video:
            # Make sure the frame we want is in the cache.  If not,
            # expand a segment of the video starting from here to
            # acquire it.
            _, exists = cache.lookup(self.frame_signature(index), frame_cache_format)
            if not exists:
                if self.decode_chunk_length is None:
                    self.explode(0, self.num_frames())
                else:
                    self.explode(max(0, index), int(self.decode_chunk_length*self.frame_rate()))

            # Return the frame from the cache.
            frame = self.get_frame_cached(index)
            assert frame is not None
            return frame
        else:
            return np.zeros([self.metrics.height, self.metrics.width, 4], np.uint8)

    def explode(self, start_index, length):
        """Exand a segment of the video into individual frames, then cache
        those frames for later."""
        assert self.has_video
        assert is_int(start_index)
        assert is_int(length)

        with temporary_current_directory():
            ffmpeg(
                f'-ss {start_index/self.frame_rate()}',
                f'-t {(length)/self.frame_rate()}',
                f'-i {self.fname}',
                f'-r {self.frame_rate()}',
                f'%06d.{frame_cache_format}',
                task=f'Exploding {os.path.basename(self.fname)}:{start_index}',
                num_frames=length
            )

            # Add each frame to the cache.
            rng = range(start_index, min(start_index+length, self.num_frames()))
            for index in rng:
                seq_fname = os.path.abspath(f'{index-start_index+1:06d}.{frame_cache_format}')
                cached_fname, exists = cache.lookup(self.frame_signature(index), frame_cache_format)
                if not exists:
                    try:
                        os.rename(seq_fname, cached_fname)
                    except FileNotFoundError:
                        # If we get here, that means ffmpeg thought a frame
                        # should exist, that ultimately was not extracted.
                        # This seems to happen from mis-estimations of the
                        # video length, or sometimes from simply missing
                        # frames.  To keep things rolling, let's fill in a
                        # black frame instead.
                        print(f"[Exploding {self.fname} did not produce frame {index}. "
                          "Using black instead.]")
                        fr = np.zeros([self.height(), self.width(), 3], np.uint8)
                        cv2.imwrite(cached_fname, fr)

                    cache.insert(cached_fname)

    def get_samples(self):
        if self.samples is None:
            if self.has_audio:
                self.samples = audio_samples_from_file(
                  self.fname,
                  self.sample_rate(),
                  self.num_channels(),
                  self.num_samples()
                )
            else:
                self.samples = np.zeros([self.metrics.num_samples(), self.metrics.num_channels])
        return self.samples


class slice_clip(MutatorClip):
    """ Extract the portion of a clip between the given times. Endpoints
    default to the start and end of the clip."""
    def __init__(self, clip, start=0, end=None):
        super().__init__(clip)
        if end is None:
            end = self.clip.length()

        require_float(start, "start time")
        require_non_negative(start, "start time")
        require_float(end, "end time")
        require_non_negative(end, "end time")
        require_less_equal(start, end, "start time", "end time")
        require_less_equal(end, clip.length(), "start time", "end time")

        self.metrics = Metrics(self.metrics, length=end-start)

        self.start_frame = int(start * self.frame_rate())
        self.start_sample = int(start * self.sample_rate())

    def frame_signature(self, index):
        return self.clip.frame_signature(self.start_frame + index)

    def get_frame(self, index):
        return self.clip.get_frame(self.start_frame + index)

    def get_samples(self):
        original_samples = self.clip.get_samples()
        return original_samples[self.start_sample:self.start_sample+self.num_samples()]


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

class reverse(MutatorClip):
    """ Reverse both the video and audio in a clip. """
    def frame_signature(self, index):
        return self.clip.frame_signature(self.num_frames() - index - 1)
    def get_frame(self, index):
        return self.clip.get_frame(self.num_frames() - index - 1)
    def get_samples(self):
        return np.flip(self.clip.get_samples(), axis=0)

class scale_volume(MutatorClip):
    """ Scale the volume of audio in a clip.  """
    def __init__(self, clip, factor):
        super().__init__(clip)
        require_float(factor, "scaling factor")
        require_positive(factor, "scaling factor")
        self.factor = factor

    def get_samples(self):
        return self.factor * self.clip.get_samples()

class crop(MutatorClip):
    """Trim the frames of a clip to show only the rectangle between
    lower_left and upper_right."""
    def __init__(self, clip, lower_left, upper_right):
        super().__init__(clip)
        require_int(lower_left[0], "lower left")
        require_positive(lower_left[0], "lower left")
        require_int(lower_left[1], "lower left")
        require_positive(lower_left[1], "lower left")
        require_int(upper_right[0], "upper right")
        require_less_equal(upper_right[0], clip.width(), "upper right", "width")
        require_int(upper_right[1], "upper right")
        require_less_equal(upper_right[1], clip.height(), "upper right", "width")
        require_less(lower_left[0], upper_right[0], "lower left", "upper right")
        require_less(lower_left[1], upper_right[1], "lower left", "upper right")

        self.lower_left = lower_left
        self.upper_right = upper_right

        self.metrics = Metrics(
          self.metrics,
          width=upper_right[0]-lower_left[0],
          height=upper_right[1]-lower_left[1]
        )

    def frame_signature(self, index):
        return ['crop', self.lower_left, self.upper_right, self.clip.frame_signature(index)]

    def get_frame(self, index):
        frame = self.clip.get_frame(index)
        ll = self.lower_left
        ur = self.upper_right
        return frame[ll[1]:ur[1], ll[0]:ur[0], :]

class draw_text(VideoClip):
    """ A clip consisting of just a bit of text. """
    def __init__(self, text, font_filename, font_size, frame_rate, length):
        super().__init__()

        require_string(font_filename, "font filename")
        require_float(font_size, "font size")
        require_positive(font_size, "font size")

        # Determine the size of the image we need.
        draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
        font = get_font(font_filename, font_size)
        size = draw.textsize(text, font=font)
        size = (size[0]+size[0]%2, size[1]+size[1]%2)

        self.metrics = Metrics(
          src=default_metrics,
          width=size[0],
          height=size[1],
          frame_rate = frame_rate,
          length=length
        )

        self.text = text
        self.font_filename = font_filename
        self.font_size = font_size
        self.frame = None

    def frame_signature(self, index):
        return ['text', self.text, self.font_filename, self.font_size,
          self.frame_rate(), self.length()]

    def get_frame(self, index):
        if self.frame is None:
            image = Image.new("RGBA", (self.width(), self.height()))
            draw = ImageDraw.Draw(image)
            draw.text(
              (0,0,),
              self.text,
              font=get_font(self.font_filename, self.font_size),
              fill=(255,0,0,255)
            )
            self.frame = np.array(image)

        return self.frame
