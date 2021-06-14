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
clip with return a new, cropped clip, but will not affect the original.

Each clip also has an audio track, which must the same length as the video
part.  There are some separate facilities for audio on its own.  Note that cv2
does not speak audio, so some of those things are delegated to ffmpeg.  You
might want to install it.

See the bottom of this file for some usage examples.

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

from abc import ABC, abstractmethod

class Clip(ABC):
    """The base class for all video clips.    A finite series of frames, each with
    identical height and width, meant to be played at a given rate, along with an
    audio clip of the same length."""

    preset = 'slow'
    bitrate = '1024k'
    cache_format = 'png'

    @abstractmethod
    def __repr__():
        """A string that describes this clip."""

    @abstractmethod
    def frame_signature(self, index):
        """A string that uniquely describes the appearance of the given frame."""

    @abstractmethod
    def frame_rate(self):
        """Frame rate of the clip, in frames per second."""

    @abstractmethod
    def width(self):
        """Width of each frame in the clip."""

    @abstractmethod
    def height(self):
        """Height of each frame in the clip."""

    @abstractmethod
    def length(self):
        """Number of frames in the clip."""

    @abstractmethod
    def get_frame(self, index):
        """Create and return one frame of the clip."""

    @abstractmethod
    def get_audio(self):
        """ Return the corresponding audio clip."""
