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
import re
import shutil
import subprocess
import tempfile
import time
import threading

import cv2
import numpy as np
import progressbar
import soundfile

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
    require(x, is_float, "positive real number", name, TypeError)

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


class FFMPEGException(Exception):
    """Raised when ffmpeg fails for some reason."""

def ffmpeg(*args, task=None, num_frames=None):
    """Run ffmpeg with the given arguments.  Optionally, maintain a progress
    bar as it goes."""

    with tempfile.NamedTemporaryFile() as stats:
        command = f"ffmpeg -y -vstats_file {stats.name} {' '.join(args)} 2> /dev/null"
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
                message = ('Alas, ffmpeg failed with return code ' +
                           f'{proc.returncode}.\nCommand was: {command}')
                # print(message)
                # print("(starting shell to examine temporary directory)")
                # os.system('bash')
                raise FFMPEGException(message)


class Metrics:
    """ A object describing the dimensions of a Clip. """
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
        if hours > 0:
            return f'{hours}:{mins:02}:{secs:02}'
        else:
            return f'{mins}:{secs:02}'

    def compute_and_cache_frame(self, index, cached_filename):
        """Call get_frame to compute one frame, and put it in the cache."""
        # Get the frame.
        frame = self.get_frame(index)

        # Make sure we got a legit frame.
        assert frame is not None, \
          "Got None instead of a real frame for " + self.frame_signature(index)
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
            return cv2.imread(cached_filename)
        else:
            # No. Generate and save to disk for next time.
            return self.compute_and_cache_frame(index, cached_filename)

    def get_cached_filename(self, index):
        """Make sure the frame is in the cache, computing it if necessary,
        and return its filename."""
        # Look for the frame we need in the cache.
        cached_filename, success = cache.lookup(
          self.frame_signature(index),
          frame_cache_format)

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
                    print(self.frame_signature(index))
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
                '-vf format=yuv420p ',
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

class VideoOnlyClip(Clip):
    """ Inherit from this for Clip classes that should use the default silent
    audio. """
    def get_samples(self):
        """Return audio samples appropriate to use as a default audio.  That
        is, silence with the appropriate metrics."""
        return np.zeros([self.metrics.num_samples(), self.metrics.num_channels])


class AudioOnlyClip(Clip):
    """ Inherit from this for Clip classes that should use the default black
    video. """
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

        self.color = [color[2], color[1], color[0], 255]
        self.frame = None

    frame_signature = AudioOnlyClip.frame_signature
    get_frame = AudioOnlyClip.get_frame
    get_samples = VideoOnlyClip.get_samples



class temporal_composite(Clip):
    """ Given a collection of (clip, start_time) tuples, form a clip
    composited from those elements as described.  When two or more
    clips overlap, the last one listed prevails."""

    def __init__(self, *args):
        super().__init__()

        self.elements = args

        # Sanity check on the inputs.
        for (i, element) in enumerate(self.elements):
            assert len(element) == 2
            (clip, start_time) = element
            require_clip(clip, f'clip {i}')
            require_float(start_time, f'start time {i}')
            require_non_negative(start_time, f'start time {i}')

        # Compute our metrics.  Same as all of the clips, except for the
        # length.
        length = 0
        for (i, element) in enumerate(self.elements):
            (clip, start_time) = element
            length = max(length, start_time + clip.length())
        self.metrics = Metrics(
          src=self.elements[0][0].metrics,
          length=length
        )

        # Check for metric mismatches.
        for (i, element) in enumerate(self.elements[1:]):
            (clip, start_time) = element
            self.metrics.verify_compatible_with(clip.metrics)

        # At each step, precompute which frame of which clip will be shown.
        self.resolved_frames = [None] * self.num_frames()
        for element in self.elements:
            (clip, start_time) = element
            start_index = int(start_time*self.frame_rate())
            for clip_index in range(clip.num_frames()):
                self.resolved_frames[start_index + clip_index] = (clip, clip_index)


    def frame_signature(self, index):
        (clip, clip_index) = self.resolved_frames[index]
        return clip.frame_signature(clip_index)

    def get_frame(self, index):
        (clip, clip_index) = self.resolved_frames[index]
        return clip.get_frame(clip_index)

    get_samples = VideoOnlyClip.get_samples


