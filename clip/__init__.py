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

from .clip import *
from .from_file import *

