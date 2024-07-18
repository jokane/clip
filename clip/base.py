""" Abstract base classes for clips. """

# pylint: disable=wildcard-import

from abc import ABC, abstractmethod
import contextlib
import os
import pprint

import cv2
import numpy as np
import soundfile

from .cache import ClipCache
from .metrics import Metrics
from .progress import custom_progressbar
from .util import temporarily_changed_directory, format_seconds_as_hms, read_image
from .validate import *


def frame_times(clip_length, frame_rate):
    """ Return the timestamps at which frames should occur for a clip of the
    given length at the given frame rate.  Specifically, generate a timestamp
    at the midpoint of the time interval for each frame. """

    frame_length = 1 / frame_rate
    t = 0.5 * frame_length

    while t <= clip_length:
        yield t
        t += frame_length

def require_clip(x, name):
    """ Raise an informative exception if x is not a Clip. """
    require(x, lambda x: isinstance(x, Clip), "Clip", name, TypeError)


class Clip(ABC):
    """The base class for all clips.  It defines a number of abstract methods
    that must be implemented in subclasses, along with a few helper methods
    that are available for all clips.

    Represents a series of frames with a certain duration, each with identical
    height and width, along with audio of the same duration.

    """

    def __init__(self):
        self.metrics = None

    @abstractmethod
    def frame_signature(self, t):
        """A string that uniquely describes the appearance of this clip at the
        given time.

        :param t: A time, in seconds.  Should be between 0 and `self.length()`.
        """

    @abstractmethod
    def request_frame(self, t):
        """Called during the rendering process, before any `get_frame()` calls,
        to indicate that a frame at the given time will be needed in the
        future.

        :param t: A time, in seconds.  Should be between 0 and `self.length()`.

        This is used to provide some advance notice to a clip that a
        `get_frame()` is coming later.  Can help if frames are generated in
        batches, such as in :class:`from_file`.

        """

    @abstractmethod
    def get_frame(self, t):
        """Create and return a frame of this clip at the given time."""

    @abstractmethod
    def get_samples(self):
        """Create and return the audio data for the clip."""

    @abstractmethod
    def get_subtitles(self):
        """Return an iterable of subtitles, each a `(start_time, end_time, text)`
        triple."""


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
        """Number of channels in the clip, i.e. `1` for mono or `2` for stereo."""
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

    def request_all_frames(self, frame_rate):
        """Submit a request for every frame in this clip.
        
        :param frame_rate: The desired frame rate, in frames per second.

        """
        fts = list(frame_times(self.length(), frame_rate))
        for t in fts:
            self.request_frame(t)

    def preview(self, frame_rate, cache_dir='/tmp/clipcache/computed'):
        """Render the video part and display it in a window on screen.

        :param frame_rate: The desired frame rate, in frames per second.
        :param cache_dir: The directory to use for the frame cache.

        """
        cache = ClipCache(cache_dir)

        with custom_progressbar("Previewing", self.length()) as pb:
            self.request_all_frames(frame_rate)
            for t in frame_times(self.length(), frame_rate):
                frame = self.get_frame_cached(cache, t)
                pb.update(t)
                cv2.imshow("", frame)
                cv2.waitKey(1)

        cv2.destroyWindow("")

    def verify(self, frame_rate, verbose=False):
        """Call the appropriate methods to fully realize this clip, checking
        that the right sizes and formats of images are returned by
        `get_frame()`, the right length of format of audio is returned by
        `get_samples()`, and the right kinds of subtitles are returned by
        `get_subtitles()`.

        Useful for debugging and testing.

        :param frame_rate: The desired frame rate, in frames per second.
        :param verbose: Set this to `True` to get lots of diagnostic output.
        """

        self.metrics.verify()

        require_positive(frame_rate, 'frame rate')

        for t in frame_times(self.length(), frame_rate):
            self.request_frame(t)

        for t in frame_times(self.length(), frame_rate):
            sig = self.frame_signature(t)
            if verbose:
                print(f'{t:0.2f}', end=" ")
                pprint.pprint(sig)
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

        subtitles = self.get_subtitles()
        for subtitle in subtitles:
            assert len(subtitle) == 3
            assert is_non_negative(subtitle[0])
            assert is_non_negative(subtitle[1])
            assert subtitle[0] < subtitle[1]
            assert is_string(subtitle[2])

    def stage(self, directory, cache, frame_rate, filename=""):
        """Get everything for this clip onto to disk in the specified
        directory:

            - For each frame, a symlink into a cache directory, named in numerical order.
            - FLAC file of the audio called `audio.flac`
            - Subtitles as an SRT file call `subtitles.srt`

        :param directory: The directory in which to stage things.
        :param cache: A :class:`ClipCache` to use to get the frames, or to
                store the frames if they need to be generated.
        :param frame_rate: Output frame rate in frames per second.
        :param filename: An optional name for the file to which the staged
                frames will be saved.  Used here only to make the progress bar
                more informative.

        """

        # Do things in the requested directory.
        with temporarily_changed_directory(directory):
            # Audio.
            audio_fname = 'audio.flac'
            data = self.get_samples()
            assert data is not None
            soundfile.write(audio_fname, data, self.sample_rate())

            # Subtitles
            subtitles_fname = 'subtitles.srt'
            self.save_subtitles(subtitles_fname)

            # Video.
            task = f"Staging {filename}" if filename else "Staging"
            self.request_all_frames(frame_rate)

            fts = list(frame_times(self.length(), frame_rate))
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

    def save_subtitles(self, destination):
        """Save the subtitles for this clip to the given file.

        :param destination: A string filename or file-like object telling
                             where to send the subtitles.

        """

        with contextlib.ExitStack() as exst:
            if isinstance(destination, str):
                f = exst.enter_context(open(destination, 'w'))
            else:
                f = destination

            for number, subtitle in enumerate(self.get_subtitles()):
                print(number+1, file=f)
                hms0 = format_seconds_as_hms(subtitle[0])
                hms1 = format_seconds_as_hms(subtitle[1])
                print(hms0, '-->', hms1, file=f)
                print(subtitle[2], file=f)
                print(file=f)

    def get_cached_filename(self, cache, t):
        """Make sure the frame is in the cache given, computing it if
        necessary, and return its filename.

        :param cache: A :class:`ClipCache` to retreive or store the frame.
        :param t: A time, in seconds.  Should be between 0 and `self.length()`.

        :return: The full path to a file containing the frame at time `t`.

        """
        # Look for the frame we need in the cache.
        cached_filename, success = cache.lookup(self.frame_signature(t),
                                                cache.frame_format)

        # If it wasn't there, generate it and save it there.
        if not success:
            self.compute_and_cache_frame(t, cache, cached_filename)

        # Done!
        return cached_filename

    def compute_and_cache_frame(self, t, cache, cached_filename):
        """Call `get_frame()` to compute one frame, save it to a file, and note
        in the cache that this new file now exists.

        :param t: A time, in seconds.  Should be between 0 and `self.length()`.
        :param cache: A :class:`ClipCache` to store the frame after computing it.
        :param cached_filename: The full path, within the cache directory,
                where the frame should be saved.

        """
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
        if needed.

        :param t: A time, in seconds.  Should be between 0 and `self.length()`.
        :param cache: A :class:`ClipCache` in which to look, and in which to
                store the frame if it needs to be computed.

        """
        # Look for the frame we need in the cache.
        cached_filename, success = cache.lookup(self.frame_signature(t),
                                                cache.frame_format)

        # Did we find it?
        if success:
            # Yes.  Read from disk.
            return read_image(cached_filename)
        else:
            # No. Generate and save to disk for next time.
            return self.compute_and_cache_frame(t, cache, cached_filename)


class VideoClip(Clip):
    """ Inherit from this for Clip classes that really only have video, to
    default to silent audio."""
    def get_samples(self):
        """Return audio samples appropriate to use as a default audio.  That
        is, silence with the appropriate metrics."""
        return np.zeros([self.metrics.num_samples(), self.metrics.num_channels])

    def get_subtitles(self):
        return []


class AudioClip(Clip):
    """Inherit from this for Clip classes that only really have audio, to
    default to simple black frames for the video.  Then only `get_samples()` and
    `get_subtitles()` need to be defined."""
    def __init__(self):
        super().__init__()
        self.color = [0, 0, 0, 255]
        self.frame = None

    def frame_signature(self, t):
        """A signature indicating a solid black frame."""
        return ['solid', {
            'width': self.metrics.width,
            'height': self.metrics.height,
            'color': self.color
        }]

    def request_frame(self, t):
        """Does nothing."""
        pass

    def get_frame(self, t):
        """Return a solid black frame."""
        if self.frame is None:
            self.frame = np.zeros([self.metrics.height, self.metrics.width, 4], np.uint8)
            self.frame[:] = self.color
        return self.frame

class MutatorClip(Clip):
    """Inherit from this for Clip classes that modify another clip.
    Override only the parts that need to change."""
    def __init__(self, clip):
        super().__init__()
        require_clip(clip, "base clip")
        self.clip = clip
        self.metrics = clip.metrics

    def frame_signature(self, t):
        return self.clip.frame_signature(t)

    def request_frame(self, t):
        self.clip.request_frame(t)

    def get_frame(self, t):
        return self.clip.get_frame(t)

    def get_subtitles(self):
        return self.clip.get_subtitles()

    def get_samples(self):
        return self.clip.get_samples()

class FiniteIndexed:
    """Mixin for clips derived from a finite, ordered sequence of frames. Keeps
    track of a frame rat and a number of frames, and provides a method for
    converting times to frame indices."""
    def __init__(self, num_frames, frame_rate=None, length=None):

        if frame_rate is not None:
            require_float(frame_rate, "frame rate")
            require_positive(frame_rate, "frame rate")
        if length is not None:
            require_float(length, "length")
            require_positive(length, "length")

        if not frame_rate and not length:
            raise ValueError('Need either frame rate or length.')

        if frame_rate and length:
            raise ValueError('Need either frame rate or length, not both.')

        if length:
            frame_rate = num_frames/length

        self.frame_rate = frame_rate

    def time_to_frame_index(self, t):
        """Which frame would be visible at the given time?"""
        return int(t*self.frame_rate)

