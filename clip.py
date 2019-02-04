from abc import ABC, abstractmethod
import cv2
import os
import hashlib
import numpy as np
from functools import reduce

cache = dict()

class Clip(ABC):
  @abstractmethod
  def signature(index):
    """A string that uniquely identifies a given frame."""
    pass

  @abstractmethod
  def frame_rate():
    """Frame rate of the clip, in frames per second."""
    pass

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

  def get_frame(self, index):
    # Make sure we've loaded the list of cached frames.
    if not hasattr(Clip, 'cache'):
      Clip.cache = dict()
      for cached_frame in os.listdir(".cache"):
        Clip.cache[".cache/" + cached_frame] = True
      print(len(Clip.cache), "frames in the cache.")

    # Has this frame been computed before?
    sig = str(self.signature(index))
    blob = hashlib.md5(sig.encode()).hexdigest()
    cached_filename = ".cache/" + blob + ".png"
    if cached_filename in Clip.cache:
      print("[+]", sig)
      return cv2.imread(cached_filename)
    else:
      print("[ ]", sig)
      frame = self.build_frame(index)
      cv2.imwrite(cached_filename, frame)
      Clip.cache[cached_filename] = True
      return frame

  def length_secs(self):
    return self.length()/self.frame_rate()

  def play(self):
    for i in range(0, self.length()):
      frame = self.get_frame(i)
      cv2.imshow("", frame)
      cv2.waitKey(int(1000.0/self.frame_rate()))

  def save(self, fname):
    vw = cv2.VideoWriter(
      fname,
      cv2.VideoWriter_fourcc(*"mp4v"),
      self.frame_rate(),
      (self.width(), self.height())
    )
    for i in range(0, self.length()):
      frame = self.get_frame(i)
      vw.write(frame)
    vw.release()
    

class Black(Clip):
  def __init__(self, height, width, frame_rate, secs):
    self.frame_rate_ = frame_rate
    self.width_ = width
    self.height_ = height
    self.length_ = int(self.frame_rate_ * secs)

  def frame_rate(self):
    return self.frame_rate_

  def length(self):
    return self.length_

  def width(self):
    return self.width_

  def height(self):
    return self.height_

  def signature(self, index):
    return "(black:%dx%d)" % (self.width_, self.height_)

  def build_frame(self, index):
    try:
      return self.frame
    except AttributeError:
      self.frame = np.zeros([self.height_, self.width_, 3], np.uint8)
      return self.frame

class Sequence(Clip):
  def __init__(self, clips):
    self.clips = clips
    assert(len(set(map(lambda x: x.frame_rate(), self.clips))) == 1)

  def frame_rate(self):
    return self.clips[0].frame_rate()

  def width(self):
    return self.clips[0].width()

  def height(self):
    return self.clips[0].height()

  def length(self):
    return sum(map(lambda x: x.length(), self.clips))

  def signature(self, index):
    for clip in self.clips:
      if index < clip.length():
        return clip.signature(index)
      index -= clip.length()

  def build_frame(self, index):
    for clip in self.clips:
      if index < clip.length():
        return clip.get_frame(index)
      index -= clip.length()

class Slice(Clip):
  def __init__(self, clip, start_secs, end_secs):
    self.clip = clip
    self.start_frame = int(start_secs * clip.frame_rate())
    self.end_frame = int(end_secs * clip.frame_rate())

  def frame_rate(self):
    return self.clip.frame_rate()

  def width(self):
    return self.clip.width()

  def height(self):
    return self.clip.height()

  def length(self):
    return self.end_frame - self.start_frame

  def signature(self, index):
    return self.clip.signature(self.start_frame + index)

  def build_frame(self, index):
    return self.clip.get_frame(self.start_frame + index)


class VideoFile(Clip):
  def __init__(self, fname):
    self.fname = fname
    self.cap = cv2.VideoCapture(fname)
    self.last_index = -1

  def signature(self, index):
    return "(file:%s frame:%06d)" % (self.fname, index)

  def frame_rate(self):
    return float(self.cap.get(cv2.CAP_PROP_FPS))

  def width(self):
    return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

  def height(self):
    return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  def length(self):
    return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

  def build_frame(self, index):
    if index != self.last_index + 1:
      self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    self.last_index = index  
    return self.cap.read()[1]
 
class Fade(Clip):
  def __init__(self, clip1, clip2):
    self.clip1 = clip1
    self.clip2 = clip2
    assert(self.clip1.frame_rate() == self.clip2.frame_rate())
    assert(self.clip1.width() == self.clip2.width())
    assert(self.clip1.height() == self.clip2.height())
    assert(self.clip1.height() == self.clip2.height())

  def frame_rate(self):
    return self.clip1.frame_rate()

  def width(self):
    return self.clip1.width()

  def height(self):
    return self.clip1.height()

  def length(self):
    return self.clip1.length()

  def alpha(self, index):
    return (self.clip1.length()-1 - index)/(self.clip1.length()-1)

  def signature(self, index):
    a = self.alpha(index)
    return "(%f%s + %f%s)" % (a, self.clip1.signature(index), 1-a, self.clip2.signature(index))

  def build_frame(self, index):
    a = self.alpha(index)
    return cv2.addWeighted(
      self.clip1.get_frame(index), a,
      self.clip2.get_frame(index), 1.0-a,
      0
    )

def FadeSequence(clip1, clip2, overlap_secs):
  t1_secs = clip1.length_secs()-overlap_secs

  v1 = Slice(clip1, 0, t1_secs)
  v2 = Slice(clip1, t1_secs, clip1.length_secs())
  v3 = Slice(clip2, 0, overlap_secs)
  v4 = Slice(clip2, overlap_secs, clip2.length_secs())

  return Sequence([v1, Fade(v2, v3), v4])

def FadeIn(clip, fade_secs):
  black = Black(clip.height(), clip.width(), clip.frame_rate(), fade_secs)


if __name__ == "__main__":

  vid = VideoFile("/usr/local/texlive/2018/texmf-dist/tex/latex/mwe/example-movie.mp4")
  vid = Slice(vid, 0, 5)
  black = Black(vid.height(), vid.width(), vid.frame_rate(), 1)
  fade = FadeSequence(black, vid, 1)
  fade.save("test.mp4")

