"""
clip.py
--------------------------------------------------------------------------------
This is a library for manipulating and generating short video clips, such as
might be attached to a research paper.  It uses cv2 to read and write behind
the scenes, and provides abstractions for cropping, superimposing, adding text,
trimming, fading in and out, fading between clips, etc.  Additional effects can
be achieved by filtering the frames through custom functions.

The basic currency is the (abstract) Clip class, which encapsulates a sequence
of identically-sized frames, each a cv2 image, meant to be played at a certain
frame rate.  Clips may be created by reading from a video file (see:
video_file), by starting from a blank video (see: black or solid), or by using
one of the other subclasses or functions to modify existing clips.  Keep in
mind that all of these methods are non-destructive: Doing, say, a crop() on a
clip with return a new, cropped clip, but will not affect the original.

See the bottom of this file for some usage examples.

Possibly relevant implementation details:
- There is a "lazy evaluation" flavor to the execution.  Simply creating a
  clip object will check for some errors (for example, mismatched sizes or
  framerates) but will not actually do any work to produce the video.  The
  real rendering happens when one calls .save() or .play() on a clip.

- To accelerate things across multiple runs, frames are cached in a local
  directory called .cache.  This caching is done based on a string
  "signature" for each frame, which is meant to uniquely identify the visual
  contents of a frame.  For debugging purposes, each clip also has a clip
  "signature", meant to describe the composition of the clip in a
  human-readable way.

- Some parts things are subclasses of Clip, whereas other parts are just
  functions.  However, all of them use snake_case because it should not matter
  to a user whether it's a subclass or a function.

--------------------------------------------------------------------------------

"""

from PIL import Image, ImageFont, ImageDraw
from abc import ABC, abstractmethod
import cv2
import os
import hashlib
import numpy as np
import glob
from functools import reduce

def isfloat(x):
  try:
    float(x)
    return True
  except TypeError:
    return False

def isiterable(x):
  try:
    iter(x)
    return True
  except TypeError:
    return False

class Clip(ABC):
  """The base class for everything else.  A finite series of frames, each with
  identical height and width, meant to be played at a given rate."""
  @abstractmethod
  def signature():
    """A string that describes this clip."""
    pass

  @abstractmethod
  def frame_signature(index):
    """A string that uniquely describes the appearance of the given frame."""
    pass

  @abstractmethod
  def frame_rate():
    """Frame rate of the clip, in frames per second."""
    pass

  def fr(self):
    """Same as frame_rate(), but shorter."""
    return self.frame_rate()

  @abstractmethod
  def width():
    """Width of each frame in the clip."""
    pass

  @abstractmethod
  def height():
    """Height of each frame in the clip."""
    pass

  @abstractmethod
  def length():
    """Number of frames in the clip."""
    pass

  @abstractmethod
  def get_frame(self, index):
    """Create and return one frame of the clip."""
    pass

  def get_frame_cached(self, index):
    """Return one frame of the clip, from the cache if possible, but from scratch if necessary."""

    # Make sure we've loaded the list of cached frames.
    if not hasattr(Clip, 'cache'):
      Clip.cache = dict()
      try:
        for cached_frame in os.listdir(".cache"):
          Clip.cache[".cache/" + cached_frame] = True
      except FileNotFoundError:
        os.mkdir(".cache")

      print(len(Clip.cache), "frames in the cache.")

    # Has this frame been computed before?
    sig = str(self.frame_signature(index))
    blob = hashlib.md5(sig.encode()).hexdigest()
    cached_filename = ".cache/" + blob + ".png"
    if cached_filename in Clip.cache:
      # Yes.  Read it from the cache.
      print("[+]", sig)
      frame = cv2.imread(cached_filename)
    else:
      # No.  Compute it directly.
      print("[ ]", sig)
      frame = self.get_frame(index)
      cv2.imwrite(cached_filename, frame)
      Clip.cache[cached_filename] = True

    # Some sanity checks.
    assert frame is not None, "Got None instead of a real frame for " + sig
    assert frame.shape[0] == self.height(), "From %s, I got a frame of height %d instead of %d." % (self.frame_signature(index), frame.shape[0], self.height())
    assert frame.shape[1] == self.width(), "For %s, I got a frame of width %d instead of %d." % (self.frame_signature(index), frame.shape[1], self.width())

    # Done!
    return frame

  def length_secs(self):
    """Return the length of the clip, in seconds."""
    return self.length()/self.frame_rate()

  def play(self, keep_frame_rate=True):
    """Render the video and display it in a window on screen."""
    self.realize(keep_frame_rate=keep_frame_rate, play=True)

  def save(self, fname):
    """Render the video and save it as an MP4 file."""
    self.realize(save_fname=fname)

  def saveplay(self, fname, keep_frame_rate=True):
    """Play and save at the same time."""
    self.realize(save_fname=fname, play=True, keep_frame_rate=keep_frame_rate)
  
  def realize(self, save_fname=None, play=False, keep_frame_rate=True):
    """Main function for saving and playing, or both."""
    if save_fname:
      vw = cv2.VideoWriter(
        save_fname,
        cv2.VideoWriter_fourcc(*"mp4v"),
        self.frame_rate(),
        (self.width(), self.height())
      )

    try: 
      for i in range(0, self.length()):
        frame = self.get_frame_cached(i)
        if save_fname:
          vw.write(frame)
        if play:
          cv2.imshow("", frame)
          if keep_frame_rate:
            cv2.waitKey(int(1000.0/self.frame_rate()))
          else:
            cv2.waitKey(1)
    finally:
      if save_fname:
        vw.release()
    
class solid(Clip):
  """A clip with the specified size, frame rate, and length, in which each frame has the same solid color."""
  def __init__(self, height, width, frame_rate, length, color):
    # Sanity check.
    assert isinstance(height, int)
    assert height > 0

    assert isinstance(width, int)
    assert width > 0

    assert isfloat(frame_rate)
    assert frame_rate > 0

    assert isfloat(length)
    assert length > 0

    assert len(color) == 3
    assert isinstance(color[0], int)
    assert isinstance(color[1], int)
    assert isinstance(color[2], int)
    assert color[0] >= 0 and color[0] < 256
    assert color[1] >= 0 and color[1] < 256
    assert color[2] >= 0 and color[2] < 256

    # Remember the details.
    self.frame_rate_ = frame_rate
    self.width_ = width
    self.height_ = height
    self.length_ = int(length)
    self.color = color

  def frame_rate(self):
    return self.frame_rate_

  def length(self):
    return self.length_

  def width(self):
    return self.width_

  def height(self):
    return self.height_

  def signature(self):
    return "(solid:%s, %dx%d, %d frames)" % (self.color, self.width_, self.height_, self.length_)

  def frame_signature(self, index):
    return "(solid: %s, %dx%d)" % (self.color, self.width_, self.height_)

  def get_frame(self, index):
    try:
      return self.frame
    except AttributeError:
      self.frame = np.zeros([self.height_, self.width_, 3], np.uint8)
      self.frame[:] = self.color
      return self.frame

def black(height, width, frame_rate, length):
  """A clip with of solid black frames."""
  return solid(height, width, frame_rate, length, (0,0,0))

def white(height, width, frame_rate, length):
  """A clip with of white black frames."""
  return solid(height, width, frame_rate, length, (255,255,255))


class repeat_frame(Clip):
  """A clip that shows the same frame, from another clip, over and over."""
  def __init__(self, clip, frame_index, length):
    assert isinstance(clip, Clip)

    assert isinstance(frame_index, int)
    assert frame_index >= 0
    assert frame_index < clip.length(), "Trying to repeat frame %d of %s, but the last valid frame index is %s." % (frame_index, clip.signature(), clip.length()-1)

    assert isinstance(length, int)
    assert length > 0

    self.clip = clip
    self.frame_index = frame_index
    self.length_ = length

  def frame_rate(self):
    return self.clip.frame_rate()

  def length(self):
    return self.length_

  def width(self):
    return self.clip.width()

  def height(self):
    return self.clip.height()

  def signature(self):
    return "repeat frame(%d of %s, %d frames)" % (self.frame_index, self.clip.signature(), self.length_)

  def frame_signature(self, index):
    return self.clip.frame_signature(self.frame_index)

  def get_frame(self, index):
    return self.clip.get_frame(self.frame_index)

class filter_frames(Clip):
  """A clip formed by passing the frames of another clip through some function.
  The function should have one argument, the input frame, and return the output
  frame.  The output frames may have a different size from the input ones, but
  must all be the same size across the whole clip."""

  def __init__(self, clip, func, name=None):
    assert isinstance(clip, Clip)
    assert callable(func)
    self.clip = clip
    self.func = func
    if name:
      self.func.__name__ = name
    self.sample_frame = func(clip.get_frame(0))

  def frame_rate(self):
    return self.clip.frame_rate()

  def length(self):
    return self.clip.length()

  def width(self):
    return self.sample_frame.shape[1]

  def height(self):
    return self.sample_frame.shape[0]

  def signature(self):
    return "%s(%s)" % (self.func.__name__, self.clip.signature())

  def frame_signature(self, index):
    return "%s(%s)" % (self.func.__name__, self.clip.frame_signature(index))

  def get_frame(self, index):
    return self.func(self.clip.get_frame(index))

def crop(clip, lower_left, upper_right):
  """Trim the frames of a clip to show only the rectangle between lower_left and upper_right."""
  assert isinstance(clip, Clip)

  assert isinstance(lower_left[0], int)
  assert lower_left[0] >= 0

  assert isinstance(lower_left[1], int)
  assert lower_left[1] >= 0

  assert isinstance(upper_right[0], int)
  assert upper_right[0] < clip.width()

  assert isinstance(upper_right[1], int)
  assert upper_right[1] < clip.height()

  assert lower_left[0] < upper_right[0]
  assert lower_left[1] < upper_right[1]

  def crop_filter(frame):
    return frame[lower_left[1]:upper_right[1], lower_left[0]:upper_right[0], :]

  return filter_frames(clip, crop_filter, name=f'crop{lower_left}{upper_right}')

def scale_by_factor(clip, factor):
  """Scale the frames of a clip by a given factor."""
  assert isinstance(clip, Clip)
  assert isfloat(factor)
  assert factor > 0
  new_width = int(factor * clip.width())
  new_height = int(factor * clip.height())
  return scale_to_size(clip, new_width, new_height)

def scale_to_size(clip, new_width, new_height):
  """Scale the frames of a clip by a given factor."""
  assert isinstance(clip, Clip)
  def scale_filter(frame):
    return cv2.resize(frame, (new_width, new_height))
  scale_filter.__name__ = f'scale[{new_width}x{new_height}]'
  return filter_frames(clip, scale_filter)

class chain(Clip):
  """Concatenate a series of clips, which must all have the same frame size and frame rate."""

  def __init__(self, *args):
    for clip in args: assert isinstance(clip, Clip)
    assert len(set(map(lambda x: x.width(), args))) == 1, "Cannot chain clips because the widths do not match." + str(list(map(lambda x: x.width(), self.clips)))
    assert len(set(map(lambda x: x.height(), args))) == 1, "Cannot chain clips because the heights do not match." + str(list(map(lambda x: x.heights(), self.clips)))
    assert len(set(map(lambda x: x.frame_rate(), args))) == 1, "Cannot chain clips because the framerates do not match." + str(list(map(lambda x: x.frame_rate(), self.clips)))
    self.clips = args

  def frame_rate(self):
    return self.clips[0].frame_rate()

  def width(self):
    return self.clips[0].width()

  def height(self):
    return self.clips[0].height()

  def length(self):
    return sum(map(lambda x: x.length(), self.clips))

  def signature(self):
    return "(chain " + " then ".join(list(map(lambda x: x.signature(), self.clips))) + ")"

  def frame_signature(self, index):
    for clip in self.clips:
      if index < clip.length():
        return clip.frame_signature(index)
      index -= clip.length()

  def get_frame(self, index):
    for clip in self.clips:
      if index < clip.length():
        return clip.get_frame(index)
      index -= clip.length()
    assert False

def slice_by_secs(clip, start_secs, end_secs):
  """Extract the portion of a clip between the given times, which are specified in seconds."""
  assert isinstance(clip, Clip)
  assert isfloat(start_secs)
  assert isfloat(end_secs)
  assert start_secs < end_secs
  assert start_secs >= 0
  return slice_by_frames(clip, int(start_secs * clip.frame_rate()), int(end_secs * clip.frame_rate()))

class slice_by_frames(Clip):
  """Extract the portion of a clip between the given times, which are specified in frames."""
  def __init__(self, clip, start_frame, end_frame):
    assert isinstance(clip, Clip)

    assert isinstance(start_frame, int)
    assert start_frame >= 0, "Slicing %s, but slice start should be at least 0, but got %d." % (clip.signature(), start_frame)
   
    assert isinstance(end_frame, int)
    assert end_frame <= clip.length(), "Slicing %s, but slice end %d is beyond the end of the clip (%d)" % (clip.signature(), end_frame, clip.length())

    assert start_frame <= end_frame, "Slicing %s, but slice end %d is before slice start %d" % (clip.signature(), end_frame, start_frame)

    self.clip = clip
    self.start_frame = start_frame
    self.end_frame = end_frame

  def frame_rate(self):
    return self.clip.frame_rate()

  def width(self):
    return self.clip.width()

  def height(self):
    return self.clip.height()

  def length(self):
    return self.end_frame - self.start_frame

  def signature(self):
    return "(%s from %d to %d)" % (self.clip.signature(), self.start_frame, self.end_frame)

  def frame_signature(self, index):
    return self.clip.frame_signature(self.start_frame + index)

  def get_frame(self, index):
    return self.clip.get_frame(self.start_frame + index)


class video_file(Clip):
  """Read a clip from a file."""
  def __init__(self, fname):
    assert isinstance(fname, str)
    assert os.path.isfile(fname), f'Trying to open {fname}, which does not exist.'

    self.fname = fname
    self.cap = cv2.VideoCapture(fname)
    self.last_index = -1
    assert self.frame_rate() > 0, "Frame rate is 0 for %s.  Problem opening the file?" % fname

  def signature(self):
    return self.fname

  def frame_signature(self, index):
    return "%s:%06d" % (self.fname, index)

  def frame_rate(self):
    return float(self.cap.get(cv2.CAP_PROP_FPS))

  def width(self):
    return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

  def height(self):
    return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  def length(self):
    return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

  def get_frame(self, index):
    assert index < self.length(), "Requesting frame %d from %s, which only has %d frames." % (index, self.fname, self.length())
    if index != self.last_index + 1:
      self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    self.last_index = index  
    frame = self.cap.read()[1]
    if frame is None:
      raise Exception("When reading a frame from %s, got None instead of a frame." % self.fname)
    return frame
 
class fade(Clip):
  """Fade between two clips, which must be equal in length, frame size, and frame rate."""
  def __init__(self, clip1, clip2):
    assert isinstance(clip1, Clip)
    assert isinstance(clip2, Clip)
    assert clip1.frame_rate() == clip2.frame_rate(), "Mismatched frame rates %d and %d" % (self.clip1.frame_rate(), self.clip2.frame_rate())
    assert clip1.width() == clip2.width()
    assert clip1.height() == clip2.height()

    self.clip1 = clip1
    self.clip2 = clip2

  def frame_rate(self):
    return self.clip1.frame_rate()

  def width(self):
    return self.clip1.width()

  def height(self):
    return self.clip1.height()

  def length(self):
    return self.clip1.length()

  def alpha(self, index):
    return (self.clip1.length()-1 - index)/(self.clip1.length())

  def signature(self):
    return "fade(%s, %s)" % (self.clip1.signature(), self.clip2.signature())

  def frame_signature(self, index):
    a = self.alpha(index)
    return "(%f%s + %f%s)" % (a, self.clip1.frame_signature(index), 1-a, self.clip2.frame_signature(index))

  def get_frame(self, index):
    a = self.alpha(index)
    return cv2.addWeighted(
      self.clip1.get_frame(index), a,
      self.clip2.get_frame(index), 1.0-a,
      0
    )



class add_text(Clip):
  """Superimpose text onto every frame of a clip.  The 'text' parameter
  should be iterable, with each element consisting of a font size, the filename
  of a TrueType font, and the actual text to draw.  Text starts
  at the top and move downward with each line.  White and centered by default."""

  def __init__(self, clip, text_list, align='center', color='white'):
    assert isinstance(clip, Clip)
    assert isiterable(text_list)
    self.clip = clip
    self.text_list = text_list
    self.align = align

    # TODO: Colors are not handled correctly here, almost certainly because
    # of RGB/BGR/etc nonsense.  Might need to transpose things when we
    # convert back to a numpy array.
    self.color = color

  def frame_rate(self):
    return self.clip.frame_rate()

  def width(self):
    return self.clip.width()

  def height(self):
    return self.clip.height()

  def length(self):
    return self.clip.length()

  def signature(self):
    return "(%s+text(%s)" % (self.clip.signature(), self.text_list)

  def frame_signature(self, index):
    return "(%s+text[%s/%s](%s)" % (self.clip.frame_signature(index), self.align, self.color, self.text_list)

  def get_frame(self, index):
    frame = self.clip.get_frame(index)
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    y = 25
    for (points, font_file, text) in self.text_list:
      font = ImageFont.truetype(font_file, points)
      size = draw.textsize(text, font=font)
      if self.align == 'center':
        draw.text((self.width()/2 - size[0]/2, y), text, font=font, fill=self.color)
      elif self.align == 'left':
        draw.text((0, y), text, font=font, fill=self.color)

      y += 1.2*points

    array = np.array(pil_image)
    return array

def fade_chain(clip1, clip2, overlap_frames):
  """Concatenate two clips, with some fading overlap between them."""
  assert isinstance(clip1, Clip)
  assert isinstance(clip2, Clip)
  assert isfloat(overlap_frames)
  assert overlap_frames >= 0
  assert overlap_frames <= clip1.length()
  assert overlap_frames <= clip2.length()
  overlap_frames = int(overlap_frames)

  t1_frames = clip1.length()-overlap_frames

  v1 = slice_by_frames(clip1, 0, t1_frames)
  v2 = slice_by_frames(clip1, t1_frames, clip1.length())
  v3 = slice_by_frames(clip2, 0, overlap_frames)
  v4 = slice_by_frames(clip2, overlap_frames, clip2.length())

  return chain(v1, fade(v2, v3), v4)

def fade_in(clip, fade_frames):
  """Fade in from black."""
  assert isinstance(clip, Clip)
  assert isfloat(fade_frames)
  assert fade_frames >= 0
  fade_frames = int(fade_frames)
  assert fade_frames <= clip.length(), "Cannot fade into %s for %f frames, because the clip is only %f frames long." % (clip.signature(), fade_frames, clip.length())

  blk = black(clip.height(), clip.width(), clip.frame_rate(), fade_frames)
  return fade_chain(blk, clip, fade_frames)

def fade_out(clip, fade_frames):
  """Fade out to black."""
  assert isinstance(clip, Clip)
  assert isfloat(fade_frames)
  assert fade_frames >= 0
  fade_frames = int(fade_frames)
  assert fade_frames <= clip.length(), "Cannot fade out from %s for %f frames, because the clip is only %f frames long." % (clip.signature(), fade_frames, clip.length())

  blk = black(clip.height(), clip.width(), clip.frame_rate(), fade_frames)
  return fade_chain(clip, blk, fade_frames)

class superimpose(Clip):
  """Superimpose one clip on another, at a given place in each frame, starting
  at a given time."""

  def __init__(self, under_clip, over_clip, x, y, start_frame):
    assert isinstance(under_clip, Clip)
    assert isinstance(over_clip, Clip)
    assert isinstance(x, int)
    assert x >= 0
    assert isinstance(y, int)
    assert y >= 0
    assert isfloat(start_frame)
    assert start_frame >= 0
    start_frame = int(start_frame)

    assert y + over_clip.height() <= under_clip.height(), "Superimposing %s onto %s at (%d, %d), but the under clip is not tall enough.  It would need to be %d, but is only %d." % (over_clip.signature(), under_clip.signature(), x, y, y + over_clip.height(), under_clip.height())
    assert x + over_clip.width() <= under_clip.width(), "Superimposing %s onto %s at (%d, %d), but the under clip is not wide enough." % (over_clip.signature(), under_clip.signature(), x, y)
    assert start_frame + over_clip.length() <= under_clip.length(), "Superimposing %s onto %s at frame %d, but the under clip is not long enough." % (over_clip.signature(), under_clip.signature(), start_frame)
    assert under_clip.frame_rate() == over_clip.frame_rate(), "Superimposing %s onto %s at frame %d, but the framerates do not match." % (over_clip.signature(), under_clip.signature())

    self.under_clip = under_clip
    self.over_clip = over_clip
    self.x = x
    self.y = y
    self.start_frame = start_frame


  def frame_rate(self):
    return self.under_clip.frame_rate()

  def width(self):
    return self.under_clip.width()

  def height(self):
    return self.under_clip.height()

  def length(self):
    return self.under_clip.length()

  def signature(self):
    return "%s+(%s@(%d,%d,%d))" % (self.under_clip.signature(), self.over_clip.signature(), self.x, self.y, self.start_frame)

  def frame_signature(self, index):
    if index >= self.start_frame and index - self.start_frame < self.over_clip.length():
      return "%s+(%s@(%d,%d,%d))" % (self.under_clip.frame_signature(index), self.over_clip.frame_signature(index), self.x, self.y, self.start_frame)
    else:
      return self.under_clip.frame_signature(index)

  def get_frame(self, index):
    frame = np.copy(self.under_clip.get_frame(index))
    if index >= self.start_frame and index - self.start_frame < self.over_clip.length():
      x0 = self.x
      x1 = self.x + self.over_clip.width()
      y0 = self.y
      y1 = self.y + self.over_clip.height()
      frame[y0:y1, x0:x1, :] = self.over_clip.get_frame(index - self.start_frame)
    return frame

def superimpose_center(under_clip, over_clip, start_frame):
  """Superimpose one clip on another, in the center of each frame, starting at
  a given time."""
  assert isinstance(under_clip, Clip)
  assert isinstance(over_clip, Clip)
  assert isinstance(start_frame, int)
  x = int(under_clip.width()/2) - int(over_clip.width()/2)
  y = int(under_clip.height()/2) - int(over_clip.height()/2)
  return superimpose(under_clip, over_clip, x, y, start_frame)

class image_glob(Clip):
  """Form a video from a collection of identically-sized image files that match a unix-style pattern, at a given frame rate."""

  def __init__(self, pattern, frame_rate):
    self.pattern = pattern
    assert isfloat(frame_rate)
    assert frame_rate > 0

    self.frame_rate_ = frame_rate
    self.filenames = sorted(glob.glob(pattern))
    assert len(self.filenames) > 0, "No files matched pattern: " + pattern
    self.sample_frame = cv2.imread(self.filenames[0])
    assert self.sample_frame is not None

  def frame_rate(self):
    return self.frame_rate_

  def width(self):
    return self.sample_frame.shape[1]

  def height(self):
    return self.sample_frame.shape[0]

  def length(self):
    return len(self.filenames)

  def signature(self):
    return "%s@%dfps" % (self.pattern, self.frame_rate())

  def frame_signature(self, index):
    return self.filenames[index]

  def get_frame(self, index):
    return cv2.imread(self.filenames[index])


class force_frame_rate(Clip):
  """Change the frame rate at which a clip thinks it should be played."""
  def __init__(self, clip, frame_rate):
    assert isinstance(clip, Clip)
    assert isfloat(frame_rate)
    assert frame_rate > 0

    self.clip = clip
    self.frame_rate_ = frame_rate

  def frame_rate(self):
    return self.frame_rate_

  def signature(self):
    return "%s@%sfps" % (self.clip.signature(), self.frame_rate_)

  def width(self): return self.clip.width()
  def height(self): return self.clip.height()
  def length(self): return self.clip.length()
  def get_frame(self, index): return self.clip.get_frame(index)
  def frame_signature(self, index): return self.clip.frame_signature(index)




if __name__ == "__main__":
  # Some basic tests/illustrations.  The source font and video are part of
  # texlive, which might be installed on your computer already.

  font_filename = "/usr/local/texlive/2019/texmf-dist/fonts/truetype/sorkin/merriweather/MerriweatherSans-Regular.ttf"
  video_filename = "/usr/local/texlive/2019/texmf-dist/tex/latex/mwe/example-movie.mp4"

  vid = video_file(video_filename)
  vid = slice_by_secs(vid, 0, 7)

  small_vid = slice_by_frames(filter_frames(vid, lambda x: cv2.resize(x, (150, 100))), 0, 100)

  vid = superimpose(vid, small_vid, 200, 100, 30)
  vid = superimpose_center(vid, small_vid, 100)

  title = add_text(black(vid.height(), vid.width(), vid.frame_rate(), 5*vid.frame_rate()), [
    (70, font_filename, "Test Video for clip.py"),
    (10, font_filename, "If you can read this, you don't need glasses.")
  ])
  title = fade_in(title, 0.5*vid.frame_rate())
  title = fade_out(title, 0.5*vid.frame_rate())

  fade = fade_chain(title, vid, 1)

  print(fade.signature())
  fade.play()

