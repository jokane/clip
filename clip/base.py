""" Abstract base classes for clips. """

# pylint: disable=wildcard-import

from abc import ABC, abstractmethod
import os
import pprint
import re
import sys
import tempfile

import cv2
import numpy as np
import soundfile

from .cache import ClipCache
from .ffmpeg import ffmpeg
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
    def request_frame(self, t):
        """Called during the rendering process, before any get_frame calls, to
        indicate that a frame at the given t will be needed in the future.  Can
        help if frames are generated in batches, such as in from_file."""

    @abstractmethod
    def get_frame(self, t):
        """Create and return a frame of this clip at the given time."""

    @abstractmethod
    def get_samples(self):
        """Create and return the audio data for the clip."""

    @abstractmethod
    def get_subtitles(self):
        """Return an iterable of subtitles, each a (start_time, end_time, text)
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

    def request_all_frames(self, frame_rate):
        """Submit a request for every frame in this clip."""
        fts = list(frame_times(self.length(), frame_rate))
        for t in fts:
            self.request_frame(t)

    def preview(self, frame_rate, cache_dir='/tmp/clipcache/computed'):
        """Render the video part and display it in a window on screen."""
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

    def stage(self, directory, cache, frame_rate, fname=""):
        """Get everything for this clip onto to disk in the specified
        directory:  Symlinks to each frame and a flac file of the audio."""

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
            task = f"Staging {fname}" if fname else "Staging"
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

    def save(self, fname, frame_rate, bitrate=None, target_size=None, two_pass=False,
             preset='slow', cache_dir='/tmp/clipcache/computed', burn_subtitles=False):
        """ Save to a file.

        Bitrate controls the target bitrate.  Handles the tradeoff between
        file size and output quality.

        Target size specifies the file size we want, in MB.

        At most one of bitrate and target_size should be given.  If both are
        omitted, the default is to target a bitrate of 1024k.

        Preset controls how quickly ffmpeg encodes.  Handles the tradeoff
         encoding speed and output quality.  Choose from:
            ultrafast superfast veryfast faster fast medium slow slower veryslow
        The documentation for these says to "use the slowest preset you have
        patience for." """

        # First, a simple case: If we're saving to an audio-only format, it's
        # easy.
        if re.search('.(flac|wav)$', fname):
            assert not burn_subtitles
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

        # Assemble all of the parts.
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

                subtitles_exist = os.stat('subtitles.srt').st_size > 0
                if subtitles_exist:
                    subtitles_input='-i subtitles.srt -c:s mov_text -metadata:s:s:0 language=eng'
                else:
                    subtitles_input=''
                if subtitles_exist and burn_subtitles:
                    subtitles_filter = ',subtitles=subtitles.srt'
                else:
                    subtitles_filter = ''
                args = [
                    f'-framerate {frame_rate}',
                    f'-i %06d.{cache.frame_format}',
                    '-i audio.flac',
                    subtitles_input,
                    '-vcodec libx264',
                    '-f mp4',
                    f'-vb {bitrate}' if bitrate else '',
                    f'-preset {preset}' if preset else '',
                    '-profile:v high',
                    f'-filter_complex "format=yuv420p,pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2,fps={frame_rate}{subtitles_filter}"', #pylint: disable=line-too-long
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

    def save_subtitles(self, filename):
        """Save the subtitles for this clip to the given file."""
        with open(filename, 'w') as f:
            for number, subtitle in enumerate(self.get_subtitles()):
                print(number+1, file=f)
                hms0 = format_seconds_as_hms(subtitle[0])
                hms1 = format_seconds_as_hms(subtitle[1])
                print(hms0, '-->', hms1, file=f)
                print(subtitle[2], file=f)
                print(file=f)

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
        cached_filename, success = cache.lookup(self.frame_signature(t),
                                                cache.frame_format)

        # Did we find it?
        if success:
            # Yes.  Read from disk.
            return read_image(cached_filename)
        else:
            # No. Generate and save to disk for next time.
            return self.compute_and_cache_frame(t, cache, cached_filename)

    def save_play_quit(self, frame_rate, filename="spq.mp4"): # pragma: no cover
        """ Save the video, play it, and then end the process.  Useful
        sometimes when debugging, to see a particular clip without running the
        entire program. """
        self.save(filename, frame_rate)
        os.system("mplayer " + filename)
        sys.exit(0)

    spq = save_play_quit


class VideoClip(Clip):
    """ Inherit from this for Clip classes that really only have video, to
    default to silent audio. """
    def get_samples(self):
        """Return audio samples appropriate to use as a default audio.  That
        is, silence with the appropriate metrics."""
        return np.zeros([self.metrics.num_samples(), self.metrics.num_channels])

    def get_subtitles(self):
        return []


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

    def request_frame(self, t):
        pass

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
