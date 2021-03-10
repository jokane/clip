"""
clip.py
--------------------------------------------------------------------------------
This is a library for manipulating and generating short video clips.  It uses
cv2 to read and write behind the scenes, and provides abstractions for
cropping, superimposing, adding text, trimming, fading in and out, fading
between clips, etc.  Additional effects can be achieved by filtering the frames
through custom functions.

The basic currency is the (abstract) Clip class, which encapsulates a sequence
of identically-sized frames, each a cv2 image, meant to be played at a certain
frame rate.  Clips may be created by reading from a video file (see:
video_file), by starting from a blank video (see: black or solid), or by using
one of the other subclasses or functions to modify existing clips.  Keep in
mind that all of these methods are non-destructive: Doing, say, a crop() on a
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
from PIL import Image, ImageFont, ImageDraw
import collections
import contextlib
import cv2
import hashlib
import itertools
import numpy as np
import os
import progressbar
import re
import scipy.signal
import soundfile
import subprocess
import sys
import tempfile
import threading
import time
import pdf2image

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

def iscolor(color):
  if len(color) != 3: return False
  if not isinstance(color[0], int): return False
  if not isinstance(color[1], int): return False
  if not isinstance(color[2], int): return False
  if color[0] < 0 or color[0] > 255: return False
  if color[1] < 0 or color[1] > 255: return False
  if color[2] < 0 or color[2] > 255: return False
  return True

def sha256sum(filename):
  # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
  h  = hashlib.sha256()
  b  = bytearray(128*1024)
  mv = memoryview(b)
  with open(filename, 'rb', buffering=0) as f:
      for n in iter(lambda : f.readinto(mv), 0):
          h.update(mv[:n])
  return h.hexdigest()

def custom_progressbar(task, steps):
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


def ffmpeg(*args, task=None, num_frames=None):
  """Run ffmpeg with the given arguments.  Optionally, maintain a progress bar
  as it goes."""

  with tempfile.NamedTemporaryFile() as stats:
    command = f"ffmpeg -y -vstats_file {stats.name} {' '.join(args)} 2> /dev/null"
    proc = subprocess.Popen(command, shell=True)

    t = threading.Thread(target=lambda : proc.communicate())
    t.start()

    if task is not None:
      with custom_progressbar(task=task, steps=num_frames) as pb:
        pb.update(0)
        while proc.poll() is None:
          try:
            with open(stats.name) as f:
              fr = int(re.findall('frame=\s*(\d+)\s', f.read())[-1])
              pb.update(min(fr, num_frames-1))
          except FileNotFoundError:
            pass
          except IndexError:
            pass
          time.sleep(1)
      
    t.join()
    
    if proc.returncode != 0:
      message = f'Alas, ffmpeg failed with return code {proc.returncode}.\nCommand was: {command}'
      print(message)
      print("(starting shell to examine temporary directory)")
      os.system('bash')
      raise Exception(message)

@contextlib.contextmanager
def temporary_current_directory():
  """Create a context in which the current directory is a new temporary
  directory.  When the context ends, the current directory is restored and the
  temporary directory is vaporized."""
  previous_current_directory = os.getcwd()

  with tempfile.TemporaryDirectory() as td:
    os.chdir(td)
    
    try:
      yield
    finally:
      os.chdir(previous_current_directory)

class ClipCache:
  """An object for managing the cache of already-computed frames and audio
  segments."""
  def __init__(self):
    self.directory = '/tmp/clipcache/'
    self.cache = None

  def scan_directory(self):
    """Examine the cache directory and remember what we see there."""
    self.cache = dict()
    try:
      for cached_frame in os.listdir(self.directory):
        self.cache[os.path.join(self.directory, cached_frame)] = True
    except FileNotFoundError:
      os.mkdir(self.directory)

    counts = '; '.join(map(lambda x: f'{x[1]} {x[0]}', collections.Counter(map(lambda x: os.path.splitext(x)[1][1:], self.cache.keys())).items()))
    print(f'Found {len(self.cache)} cached items ({counts}) in {self.directory}')


  def sig_to_fname(self, sig, ext):
    """Compute the filename where something with the given signature and
    extension should live."""
    blob = hashlib.md5(sig.encode()).hexdigest()
    return os.path.join(self.directory, f'{blob}.{ext}')

  def lookup(self, sig, ext):
    """Determine the appropriate filename for something with the given
    signature and extension.  Return a tuple with that filename followed by
    True or False, indicating whether that file exists or not."""
    if self.cache is None:
      self.scan_directory()
    cached_filename = self.sig_to_fname(sig, ext)
    return (cached_filename, cached_filename in self.cache)
  
  def insert(self, fname):
    """Update the cache to reflect the fact that the given file exists."""
    if self.cache is None:
      self.scan_directory()
    self.cache[fname] = True


cache = ClipCache()

class Clip(ABC):
  """The base class for all video clips.  A finite series of frames, each with
  identical height and width, meant to be played at a given rate, along with an
  audio clip of the same length."""

  preset = 'slow'
  bitrate = '1024k'
  cache_format = 'png'

  @abstractmethod
  def __repr__():
    """A string that describes this clip."""
    pass

  def summary(self):
    secs = self.length()/self.frame_rate()
    return f'{secs}s {self.width()}x{self.height()} {self.frame_rate()}fps'

  @abstractmethod
  def frame_signature(index):
    """A string that uniquely describes the appearance of the given frame."""
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

  @abstractmethod
  def get_frame(self, index):
    """Create and return one frame of the clip."""
    pass

  @abstractmethod
  def get_audio(self):
    """ Return the corresponding audio clip."""
    pass

  def frame_to_sample(self, index):
    assert isinstance(index, int)
    return int(index * self.get_audio().sample_rate() / self.frame_rate())

  def get_frame_cached(self, index):
    """Return one frame of the clip, from the cache if possible, but from scratch if necessary."""
    sig = self.frame_signature(index)
    cached_filename, success = cache.lookup(sig, Clip.cache_format)

    if success:
      frame = cv2.imread(cached_filename)
    else:
      frame = self.get_frame(index)
      cv2.imwrite(cached_filename, frame)
      cache.insert(cached_filename)

    # Some sanity checks.
    assert frame is not None, "Got None instead of a real frame for " + sig
    assert frame.shape[0] == self.height(), "From %s, I got a frame of height %d instead of %d." % (self.frame_signature(index), frame.shape[0], self.height())
    assert frame.shape[1] == self.width(), "For %s, I got a frame of width %d instead of %d." % (self.frame_signature(index), frame.shape[1], self.width())

    # Done!
    return frame

  def length_secs(self):
    """Return the length of the clip, in seconds."""
    return self.length()/self.frame_rate()

  def play_silently(self):
    """Render the video part and display it in a window on screen."""
    self.realize_video(keep_frame_rate=True, play=True)

  def preview(self):
    """Render the video part and display it in a window on screen."""
    self.realize_video(keep_frame_rate=False, play=True)
  
  def realize_video(self, save_fname=None, play=False, keep_frame_rate=True):
    """Main function for saving and playing, or both.  Note that the save()
    method does not use this (anymore) because cv2 cannot handle audio, so it's
    faster just to give the frames directly to ffmpeg."""
    if save_fname:
      vw = cv2.VideoWriter(
        save_fname,
        cv2.VideoWriter_fourcc(*"mp4v"),
        self.frame_rate(),
        (self.width(), self.height())
      )

    tasks = [] 
    if play and keep_frame_rate:
      tasks.append('playing')
    if play and not keep_frame_rate:
      tasks.append('previewing')
    if save_fname:
      tasks.append("saving " + save_fname)
    task = "+".join(tasks).capitalize()
    
    try:    
      with custom_progressbar(task, self.length()) as pb:
        for i in range(0, self.length()):
          frame = self.get_frame_cached(i)
          pb.update(i)
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

  def default_audio(self):
    """Utility to generate an appropriate-length silent audio clip with reasonable parameters."""
    sample_rate = 48000
    num_channels = 2
    return silence(int(self.length() * sample_rate / self.frame_rate()), sample_rate, num_channels)

  def save(self, fname):
    """Save both audio and video, in a format suitable for embedding in HTML5."""
    full_fname = os.path.join(os.getcwd(), fname)

    # Force the frame cache to be read before we change to the temp directory.
    self.get_frame_cached(0)

    with temporary_current_directory():

      audio_fname = 'audio.flac'
      self.get_audio().save(audio_fname)
      
      with custom_progressbar(f"Staging {fname}", self.length()) as pb:
        for i in range(0, self.length()):
          cached_filename, success = cache.lookup(self.frame_signature(i), Clip.cache_format)
          if not success:
            frame = self.get_frame_cached(i)
            cv2.imwrite(cached_filename, frame)
            cache.insert(cached_filename)
          os.symlink(cached_filename, f'{i:06d}.{Clip.cache_format}')
          pb.update(i)

      ffmpeg(
        f'-framerate {self.frame_rate()} -i %06d.{Clip.cache_format} -i {audio_fname} ',
        f'-vcodec libx264 -f mp4 ',
        f'-vb {Clip.bitrate}' if Clip.bitrate else '',
        f'-preset {Clip.preset}' if Clip.preset else '',
        f'-profile:v high',
        f'-vf format=yuv420p',
        f'{full_fname}',
        task=f"Encoding {fname}",
        num_frames=self.length()
      )

      asecs = self.get_audio().length()/self.get_audio().sample_rate()
      vsecs = self.length()/self.frame_rate()
      print(f'Wrote audio ({asecs:0.2f}s) and video ({vsecs:0.2f}s) to {fname}.')

  def readable_length(self):
    secs = int(self.length() / self.frame_rate())
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
      return f'{hours}:{mins:02}:{secs:02}'
    else:
      return f'{mins}:{secs:02}'


class video_file(Clip):
  """Read a video clip from a file, optionally grabbing its audio track.  If we
  don't read audio from the input file, fill in silence instead."""
  
  # Older versions of this used cv2.VideoCapture.  But that seemed to be buggy
  # on some types of videos, and needed ffmpeg hacks to get the audio anyway.
  # This new version uses ffmpeg to explode the video directly into the cache.

  def __init__(self, fname, audio=True, decode_chunk_size=1000, forced_length=None):
    assert isinstance(fname, str)
    assert os.path.isfile(fname), f'Trying to open {fname}, which does not exist.'

    self.fname = os.path.abspath(fname)

    self.acquire_dimensions(forced_length=forced_length)

    self.decode_chunk_size = decode_chunk_size

  def __init__(self, fname, audio=True):
    assert isinstance(fname, str)
    assert os.path.isfile(fname), f'Trying to open {fname}, which does not exist.'

    self.fname = fname
    self.cap = cv2.VideoCapture(fname)
    self.cap.setExceptionMode(True)
    self.last_index = -1
    assert self.frame_rate() > 0, "Frame rate is 0 for %s.  Problem opening the file?" % fname

    if audio:
      # Grab the audio.
      self.audio = audio_file(fname)

      # Make sure it has the length we expect.  Small differences seem to be
      # normal, but complain if the difference is big.  Chop it or add silence
      # if needed.
      target_length = self.frame_to_sample(self.length())
      difference_in_samples = abs(self.audio.length() - target_length)
      difference_in_frames = difference_in_samples * self.frame_rate() / self.audio.sample_rate()
      if difference_in_frames > 1.0*self.frame_rate():
        print(f'WARNING: In {fname}, video length differs from audio length by {difference_in_frames} frames.  Correcting.', file=sys.stderr)
      self.audio = force_audio_length(self.audio, target_length)
    else:
      self.audio = self.default_audio()

    assert self.audio.length() == self.frame_to_sample(self.length())

  def acquire_dimensions(self, forced_length=None):
    # Get the width, height, length, and frame rate.
    
    # Do we have this information in the cache?
    cached_filename, exists = cache.lookup(hashlib.md5(self.fname.encode()).hexdigest(), 'dim')
    if exists:
      print(f"Using cached dimensions for {self.fname}") 
      (self.width_, self.height_, self.frame_rate_, self.length_, _) = open(cached_filename, 'r').read().split('\t')
      self.width_ = int(self.width_)
      self.height_ = int(self.height_)
      self.frame_rate_ = float(self.frame_rate_)
      self.length_ = int(self.length_)
      if forced_length:
        assert forced_length == self.length
      return
    
    
    # Here the length is computed from the reported duration; it seems
    # plausible that this might be off by a few frames, so we'll need to be
    # careful later.
    print(f"Getting dimensions for {self.fname}")
    proc = subprocess.Popen(f'ffprobe {self.fname}', shell=True, stderr=subprocess.PIPE)
    deets = proc.stderr.read().decode('utf-8')
    match = re.search(r'Stream.*Video.* (\d+)x(\d+).*, ([0-9\.]+) fps', deets)
    assert match, f"Could not find dimensions for {self.fname}" 

    self.width_ = int(match.group(1))
    self.height_ = int(match.group(2))
    self.frame_rate_ = float(match.group(3))

    if not forced_length:
      match = re.search(r'Duration: (\d\d):(\d\d):(\d\d\.\d\d)', deets)
      assert match, f"Could not find length for {fname}." 
      hours = int(match.group(1))
      minutes = int(match.group(2))
      seconds = float(match.group(3))
      self.length_ = int((60*(60*hours + minutes)+seconds) * self.frame_rate_)
    else:
      self.length_ = forced_length

    with open(cached_filename, 'w') as f:
      print('\t'.join(map(lambda x: str(x), [self.width_, self.height_, self.frame_rate_, self.length_, self.fname])), file=f)

    match = re.search(r'displaymatrix: rotation of -90.00 degrees', deets)
    if match:
      self.width_, self.height_ = self.height_, self.width_


  def __repr__(self):
    return f'video_file("{self.fname}")'

  def frame_rate(self):
    return self.frame_rate_

  def width(self):
    return self.width_

  def height(self):
    return self.height_

  def length(self):
    return self.length_

  def frame_signature(self, index):
    return "%s:%06d" % (self.fname, index)

  def get_frame(self, index):
    assert index < self.length(), "Requesting frame %d from %s, which only has %d frames." % (index, self.fname, self.length())
    
    # Make sure it's in the cache.  If it's not, expand a segment of the video
    # starting from here to acquire it.
    cached_filename, exists = cache.lookup(self.frame_signature(index), Clip.cache_format)
    if not exists:
      if self.decode_chunk_size is None:
        self.explode(0, self.length())
      else:
        self.explode(max(0, index-int(self.frame_rate())), self.decode_chunk_size)
    
    # Return the frame from the cache.
    return self.get_frame_cached(index)

  def explode(self, start_index, length):
    # Explode a segment of the video into individual frames, then cache those
    # frames for later.
    with temporary_current_directory():
      try:
        ffmpeg(
          f'-ss {start_index/self.frame_rate_}',
          f'-t {(length)/self.frame_rate_}',
          f'-i {self.fname}',
          f'-r {self.frame_rate_}',
          f'%06d.{Clip.cache_format}',
          task=f'Exploding {os.path.basename(self.fname)}:{start_index}',
          num_frames=length
        )
      except Exception as e:
        print(e)
        print("(starting shell to examine temporary directory)")
        os.system('bash')
        raise

      # Add each frame to the cache.
      for index in range(start_index, min(start_index+length, self.length())):
        sequential_filename = os.path.abspath(f'{index-start_index+1:06d}.{Clip.cache_format}')
        cached_filename, exists = cache.lookup(self.frame_signature(index), Clip.cache_format)
        if not exists:
          try:
            os.rename(sequential_filename, cached_filename)
          except FileNotFoundError:
            # If we get here, that means ffmpeg thought there were more
            # frames than there really were.  To keep things rolling, let's
            # fill in a black frame instead.
            print(f"[Exploding {self.fname} did not produce frame {index}.  Using black instead.]")
            fr = np.zeros([self.height_, self.width_, 3], np.uint8)
            cv2.imwrite(cached_filename, fr)

          cache.insert(cached_filename)

  def get_audio(self):
    return self.audio

class solid(Clip):
  """A video clip with the specified size, frame rate, and length, in which each frame has the same solid color."""
  def __init__(self, height, width, frame_rate, length, color):
    assert isinstance(height, int)
    assert height > 0

    assert isinstance(width, int)
    assert width > 0

    assert isfloat(frame_rate)
    assert frame_rate > 0

    assert isinstance(length, int)
    assert length > 0

    assert iscolor(color)

    self.frame_rate_ = frame_rate
    self.width_ = width
    self.height_ = height
    self.length_ = length
    self.color = [color[2], color[1], color[0]]

  def __repr__(self):
    return f'solid({self.height_}, {self.width_}, {self.frame_rate_}, {self.length_}, {self.color})'

  def frame_rate(self):
    return self.frame_rate_

  def length(self):
    return self.length_

  def width(self):
    return self.width_

  def height(self):
    return self.height_

  def frame_signature(self, index):
    return "(solid%s, %dx%d)" % (self.color, self.width_, self.height_)

  def get_frame(self, index):
    try:
      return self.frame
    except AttributeError:
      self.frame = np.zeros([self.height_, self.width_, 3], np.uint8)
      self.frame[:] = self.color
      return self.frame
  
  def get_audio(self):
    return self.default_audio()

def black(height, width, frame_rate, length):
  """A clip with of solid black frames."""
  return solid(height, width, frame_rate, length, (0,0,0))

def white(height, width, frame_rate, length):
  """A clip with of white black frames."""
  return solid(height, width, frame_rate, length, (255,255,255))


class Audio(ABC):
  """
  A base class for all audio clips.
  """

  @abstractmethod
  def __repr__(self):
    pass

  @abstractmethod
  def length(self):
    pass

  @abstractmethod
  def sample_rate(self):
    pass


  @abstractmethod
  def num_channels(self):
    pass

  @abstractmethod
  def compute_samples(self):
    pass

  def get_samples(self):
    if not hasattr(self, '_samples'):
      self._samples = self.compute_samples()
      assert len(self._samples.shape) == 2, self._samples.shape
      assert self._samples.shape[0] == self.length()
      assert self._samples.shape[1] == self.num_channels()
    return self._samples

  def save(self, fname):
    data = self.get_samples()
    assert data is not None
    soundfile.write(fname, data, self.sample_rate())

  def play(self):
    with tempfile.TemporaryDirectory() as td:
      samples = self.length()
      secs = self.length() / self.sample_rate()
      print(f'Playing audio of {samples} samples ({secs} seconds).')
      self.save(os.path.join(td, 'audio.flac'))
      os.system('mplayer ' + os.path.join(td, 'audio.flac') + ' > /dev/null 2>&1')

class audio_from_data(Audio):
  """ Form an audio clip from a numpy array. """
  def __init__(self, name, data, sample_rate):
    assert isinstance(name, str)
    assert isinstance(data, np.ndarray)
    assert isfloat(sample_rate)
    self.name = name
    self.data = data
    self.sample_rate_ = sample_rate

  def __repr__(self):
    return "audio_from_data('%s', [%d samples], %s)" % (self.name, self.data.shape[0], self.sample_rate_)

  def length(self):
    return self.data.shape[0]
    pass

  def sample_rate(self):
    return self.sample_rate_

  def num_channels(self):
    return self.data.shape[1]

  def compute_samples(self):
    return self.data
    

def audio_file(fname):
  assert isinstance(fname, str)
  assert os.path.isfile(fname), f'Trying to open {fname}, which does not exist.'

  direct_formats = list(map(lambda x: "." + x.lower(), soundfile.available_formats().keys()))
  video_formats = ['.mp4', '.mov', '.mkv', '.wmv']

  ext = os.path.splitext(fname)[1].lower()
  if ext in direct_formats:
    data, sample_rate = soundfile.read(fname, always_2d=True)
    return audio_from_data(fname, data, sample_rate)
  elif ext in video_formats: 
    cached_filename, success = cache.lookup(fname, 'flac')
    if not success:
      print(f'Extracting audio from {fname}')
      with tempfile.TemporaryDirectory() as td:
        audio_fname = os.path.join(td, 'audio.flac')
        ffmpeg(
          f'-i {fname}',
          f'-vn',
          f'{audio_fname}',
        )
        os.rename(audio_fname, cached_filename)
        cache.insert(cached_filename)
    return audio_file(cached_filename)
  else:
    raise Exception(f"Don't know how to extract audio from {ext} format: {fname}")

def silence(length, sample_rate, num_channels):
  """ Create an audio clip of the requested amount of silence. """
  return audio_from_data("silence", np.zeros((length, num_channels)), sample_rate)

class slice_audio(Audio):
  """Extract the portion of a sound between the given times, which are specified in samples."""
  def __init__(self, audio, start_sample, end_sample):
    assert isinstance(audio, Audio)
    assert isinstance(start_sample, int)
    assert isinstance(end_sample, int)
    assert start_sample >= 0, f"Slicing {audio} but slice start should be at least 0.  Got {start_sample} instead."
    assert end_sample <= audio.length(), "Slice end %d is beyond the end of the sound (%d)" % (end_sample, audio.length())

    assert start_sample <= end_sample, "Slice end %d is before slice start %d" % (end_sample, start_sample)

    self.audio = audio
    self.start_sample = start_sample
    self.end_sample = end_sample

  def __repr__(self):
    return f'slice_audio({self.audio}, {self.start_sample}, {self.end_sample})'

  def sample_rate(self):
    return self.audio.sample_rate()

  def num_channels(self):
    return self.audio.num_channels()

  def length(self):
    return self.end_sample - self.start_sample

  def compute_samples(self):
    return self.audio.get_samples()[self.start_sample:self.end_sample]


class mix(Audio):
  """
  Add together a series of audio clips, sample by sample.  The sample rates
  and number of channels must match.  The resulting length will match the
  longest of the input clips.
  """

  def __init__(self, *args):
    # Construct our list of audios.
    self.audios = list()
    for x in args:
      if isiterable(x):
        self.audios += x
      else:
        self.audios.append(x)

    # Sanity checks.
    for audio in self.audios: assert isinstance(audio, Audio)
    assert len(self.audios) > 0, "Need at least one audio to mix."
    assert len(set(map(lambda x: x.sample_rate(), self.audios))) == 1, "Cannot chain audios because the sample rates do not match." + str(list(map(lambda x: x.sample_rate(), self.audios)))
    assert len(set(map(lambda x: x.sample_rate(), self.audios))) == 1, "Cannot chain audios because the numbers of channels do not match." + str(list(map(lambda x: x.num_channels(), self.audios)))

  def __repr__(self):
    return f'mix({self.audios})'

  def sample_rate(self):
    return self.audios[0].sample_rate()

  def num_channels(self):
    return self.audios[0].num_channels()

  def length(self):
    return max(map(lambda x: x.length(), self.audios))

  def compute_samples(self):
    r = np.zeros((self.length(), self.num_channels()))
    for audio in self.audios:
      r[:audio.length()] += audio.get_samples()
    return r

class mix_at(Audio):
  """
  Add together a series of audio clips, sample by sample, each starting at
  given sample.  Arguments should be (audio, start) tuples.  The sample rates
  and number of channels must match.
  """

  def __init__(self, *args):
    for tup in args:
      assert isinstance(tup, tuple)
      audio, start = tup
      assert isinstance(audio, Audio)
      assert isinstance(start, int)

    self.tups = args

    assert len(self.tups) > 0, "Need at least one audio to mix."
    assert len(set(map(lambda x: x[0].sample_rate(), self.tups))) == 1, "Cannot mix audios because the sample rates do not match." + str(list(map(lambda x: x[0].sample_rate(), self.tups)))
    assert len(set(map(lambda x: x[0].sample_rate(), self.tups))) == 1, "Cannot mix audios because the numbers of channels do not match." + str(list(map(lambda x: x.num_channels(), self.tups)))

  def __repr__(self):
    return f'mix_at({self.tups})'

  def sample_rate(self):
    return self.tups[0][0].sample_rate()

  def num_channels(self):
    return self.tups[0][0].num_channels()

  def length(self):
    return max(map(lambda x: x[0].length() + x[1], self.tups))

  def compute_samples(self):
    r = np.zeros((self.length(), self.num_channels()))
    for audio, start in self.tups:
      r[start:start+audio.length()] += audio.get_samples()
    return r

class volume(Audio):
  """
  Scale the volume of an audio clip, sample by sample.
  """

  def __init__(self, audio, factor):
    assert isinstance(audio, Audio)
    assert isfloat(factor)
    self.audio = audio
    self.factor = factor

  def __repr__(self):
    return f'volume({self.audio}, factor)'

  def sample_rate(self):
    return self.audio.sample_rate()

  def num_channels(self):
    return self.audio.num_channels()

  def length(self):
    return self.audio.length()

  def compute_samples(self):
    return self.factor * self.audio.get_samples()

class resample(Audio):
  """
  Change both the sample rate of an audio clip and its length, using some sort
  of fancy resampling algorithm.
  """

  def __init__(self, audio, new_sample_rate, new_length):
    assert isinstance(audio, Audio)
    assert isfloat(new_sample_rate)
    assert isinstance(new_length, int)
    self.audio = audio
    self.new_sample_rate = new_sample_rate
    self.new_length = new_length

  def __repr__(self):
    return f'resample({self.audio}, {self.new_sample_rate}, {self.new_length})'

  def sample_rate(self):
    return self.new_sample_rate

  def num_channels(self):
    return self.audio.num_channels()

  def length(self):
    return self.new_length

  def compute_samples(self):
    data = self.audio.get_samples()
    x = scipy.signal.resample(data, self.new_length)
    return x

def change_sample_rate(audio, new_sample_rate):
  return resample(audio, new_sample_rate, round(audio.length() * new_sample_rate/audio.sample_rate()))
  
def timewarp_audio(audio, factor):
  return resample(audio, audio.sample_rate(), int(audio.length()*factor))
    
class chain_audio(Audio):
  """
  Concatenate a series of audio clips.  The clips may be given individually, in
  lists, or a mixture of both.

  The sample rates and number of channels must match.
  """

  def __init__(self, *args):
    # Construct our list of audios.
    self.audios = list()
    for x in args:
      if isiterable(x):
        self.audios += x
      else:
        self.audios.append(x)

    # Sanity checks.
    for audio in self.audios: assert isinstance(audio, Audio)
    assert len(self.audios) > 0, "Need at least one audio to form a chain."
    assert len(set(map(lambda x: x.sample_rate(), self.audios))) == 1, "Cannot chain audios because the sample rates do not match." + str(list(map(lambda x: x.sample_rate(), self.audios)))
    assert len(set(map(lambda x: x.num_channels(), self.audios))) == 1, "Cannot chain audios because the numbers of channels do not match." + str(list(map(lambda x: x.num_channels(), self.audios)))

  def __repr__(self):
    return f'chain_audio({self.audios})'

  def sample_rate(self):
    return self.audios[0].sample_rate()

  def num_channels(self):
    return self.audios[0].num_channels()

  def length(self):
    return sum(map(lambda x: x.length(), self.audios))

  def compute_samples(self):
    chunks = list(map(lambda x: x.get_samples(), self.audios))
    return np.concatenate(chunks)

def force_audio_length(audio, target_length):
  """Return an audio clip with exactly the given length.  Trim the end or add silence to achieve this."""
  if audio.length() < target_length:
    return chain_audio(audio, silence(target_length-audio.length(), audio.sample_rate(), audio.num_channels()))
  elif audio.length() > target_length:
    return slice_audio(audio, 0, target_length)
  else:
    return audio


class fade_audio(Audio):
  """Fade between two sounds, which must be equal in length, sample rate, and number of channels."""

  def __init__(self, audio1, audio2):
    assert isinstance(audio1, Audio)
    assert isinstance(audio2, Audio)
    assert audio1.sample_rate() == audio2.sample_rate(), "Mismatched sample rates %d and %d" % (self.audio1.sample_rate(), self.audio2.sample_rate())
    assert audio1.num_channels() == audio2.num_channels()
    assert audio1.length() == audio2.length(), f'Cannot fade audio because the lengths do not match. {audio1.length()} != {audio2.length()}'

    self.audio1 = audio1
    self.audio2 = audio2

  def __repr__(self):
    return 'fade_audio(%s, %s)' % (self.audio1.__repr__(), self.audio2.__repr__())

  def sample_rate(self):
    return self.audio1.sample_rate()

  def num_channels(self):
    return self.audio1.num_channels()

  def length(self):
    return self.audio1.length()

  def compute_samples(self):
    if self.length() > 1e8:
      print("Starting fade_audio.compute_samples on a long segment.  Hold onto your hat.")
    a1 = self.audio1.get_samples()
    a2 = self.audio1.get_samples()
    data = (
      np.linspace([1.0]*self.num_channels(), [0.0]*self.num_channels(), self.length()) * a1
      +
      np.linspace([0.0]*self.num_channels(), [1.0]*self.num_channels(), self.length()) * a2
    )
    return data

    # It's amazing how slow the naive version is...
    # a1 = self.audio1.get_samples()
    # a2 = self.audio2.get_samples()
    # r = np.zeros((self.length(), self.num_channels()))
    # for index in range(0, self.length()):
    #   a = self.alpha(index)
    #   r[index] = (1-a)*a1[index] + a*a2[index]
    # return r

class reverse_audio(Audio):
  """Same sound, played backward."""

  def __init__(self, audio):
    assert isinstance(audio, Audio)
    self.audio = audio

  def __repr__(self):
    return 'reverse_audio(%s)' % (self.audio.__repr__())

  def sample_rate(self):
    return self.audio.sample_rate()

  def num_channels(self):
    return self.audio.num_channels()

  def length(self):
    return self.audio.length()

  def compute_samples(self):
    return np.flip(self.audio.get_samples(), axis=0)


if sys.version_info.major == 3 and sys.version_info.minor < 8:
  Label = collections.namedtuple('Label', ['text', 'color', 'font', 'size', 'x', 'y', 'start', 'end'])
else:
  Label = collections.namedtuple('Label', ['text', 'color', 'font', 'size', 'x', 'y', 'start', 'end'], defaults=[None]*4)


def get_font(font, size):
  """
  Return a TrueType font for use on Pillow images, with caching to prevent
  loading the same font again and again.  (The performance improvement seems to
  be small but non-zero.)
  """
  if (font, size) not in get_font.cache:
    try:
      get_font.cache[(font, size)] = ImageFont.truetype(font, size)
    except OSError:
      raise Exception(f"Failed to open font {font}.")
  return get_font.cache[(font, size)]
get_font.cache = dict()

class add_labels(Clip):
  """
  Superimpose one or more text labels onto every frame of a clip.
  """

  def __init__(self, clip, labels):
    assert isinstance(clip, Clip)
    assert isiterable(labels)

    # If we just got one label, pretend it was a list.
    if isinstance(labels, Label):
      labels = [labels]

    for (i, label) in enumerate(labels):
      assert isinstance(label, Label)
      assert isfloat(label.size)
      assert isinstance(label.font, str)
      assert isinstance(label.text, str)
      assert isinstance(label.x, int), f'Got {type(label.x)} instead of int for label x.'
      assert isinstance(label.y, int)
      assert iscolor(label.color)

      if label.start is None:
        labels[i] = label._replace(start=0)
        label = labels[i]

      if label.end is None:
        labels[i] = label._replace(end=clip.length())
        label = labels[i]

      assert isinstance(label.start, int), f"Label start frame should be an integer, not {label.start}."
      assert isinstance(label.end, int)

    self.clip = clip
    self.labels = labels

  def __repr__(self):
    return 'add_labels(%s, %s)' % (self.clip.__repr__(), self.labels)

  def frame_rate(self):
    return self.clip.frame_rate()
  def width(self):
    return self.clip.width()
  def height(self):
    return self.clip.height()
  def length(self):
    return self.clip.length()
  def frame_signature(self, index):
    return self.clip.frame_signature(index) + "+" + ",".join(map(lambda x: f'"{x.text}" {x.color} {x.font} {x.size} {x.x} {x.y}', filter(lambda x: x.start <= index and index < x.end, self.labels)))

  def get_frame(self, index):
    frame = self.clip.get_frame(index)
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for label in self.labels:
      if label.start <= index and index < label.end:
        font = get_font(label.font, label.size)
        draw.text((label.x, label.y), label.text, font=font, fill=(label.color[2],label.color[1],label.color[0]))

    array = np.array(pil_image)
    return array

  def get_audio(self):
    return self.clip.get_audio()

def add_titles(clip, labels, x=None, y=None, halign="center", valign="center", lalign="center", spacing=1.2):
  """
  Add several labels, with the locations computed to stack nicely. The
  labels parameter should be an iterable of Labels, whose positions are
  ignored.
  """
  
  # If we just got one label, pretend it was a list.
  if isinstance(labels, Label):
    labels = [labels]
  
  # Make a dummy frame, so we can compute text sizes.
  pil_image = Image.new("RGB", (clip.width(), clip.height()))
  draw = ImageDraw.Draw(pil_image)

  # Figure out how big the rectangle containing the titles needs to be.
  width = 0
  height = 0
  for label in labels:
    font = get_font(label.font, label.size)
    size = draw.textsize(label.text, font=font)
    width = max(width, size[0])
    height += int(spacing * size[1])

  # If we didn't get an x or y position, use the center of the frame.
  if x is None: 
    x = int(clip.width()/2)
  if y is None:
    y = int(clip.height()/2)

  # Figure out where the top left corner of the box should be placed.
  if halign == 'left':
    left = x
  elif halign == 'right':
    left = x - width
  elif halign == 'center':
    left = x - int(width/2)
  else:
    raise Exception(f'Unknown halign {halign}.')

  if valign == 'top':
    top = y
  elif valign == 'bottom':
    top = y - height
  elif valign == 'center':
    top = y - int(height/2)
  else:
    raise Exception(f'Unknown valign {valign}.')

  # Create labels positioned correctly within that box.
  new_labels = list()
  y = top
  for label in labels:
    font = get_font(label.font, label.size)
    size = draw.textsize(label.text, font=font)

    if lalign == 'left':
      x = left
    elif lalign == 'right':
      x = left + width - size[0]
    elif lalign == 'center':
      x = left + int((width - size[0])/2)
    else:
      raise Exception(f'Unknown lalign {lalign}.')

    new_label = label._replace(x=x, y=y)
    new_labels.append(new_label)
    y += int(spacing * size[1])

  # Return an add_labels that uses these correctly-positioned new labels.
  return add_labels(clip, new_labels)

class fade(Clip):
  """Fade between two clips, which must be equal in length, frame size, frame rate, sample rate, and number of channels."""
  def __init__(self, clip1, clip2):
    assert isinstance(clip1, Clip)
    assert isinstance(clip2, Clip)
    assert clip1.frame_rate() == clip2.frame_rate(), "Mismatched frame rates %d and %d" % (self.clip1.frame_rate(), self.clip2.frame_rate())
    assert clip1.width() == clip2.width(), f'Mismatchjed widths {clip1.width()} and {clip2.width()}.'
    assert clip1.height() == clip2.height()
    assert clip1.length() == clip2.length()
    assert clip1.get_audio().sample_rate() == clip2.get_audio().sample_rate(), f'Cannot fade between audio with sample rate {clip1.get_audio().sample_rate()} and audio with sample rate {clip2.get_audio().sample_rate()}.'

    assert clip1.get_audio().num_channels() == clip2.get_audio().num_channels(), f'Cannot fade between audio with {clip1.get_audio().num_channels()} channel(s) and audio with {clip2.get_audio().num_channels()} channel(s).'

    assert clip1.get_audio().length() == clip2.get_audio().length(), f'Cannot fade between clips because their audio lengths do not match. {clip1.get_audio().length()} != {clip2.get_audio().length()}'

    self.clip1 = clip1
    self.clip2 = clip2
  
  def __repr__(self):
    return f'fade({self.clip1}, {self.clip2})'

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

  def get_audio(self):
    return fade_audio(self.clip1.audio, self.clip2.audio)

class chain(Clip):
  """
  Concatenate a series of clips.  The clips may be given individually, in
  lists, or a mixture of both.
  """

  def __init__(self, *args):
    # Construct our list of clips.
    self.clips = list()
    for x in args:
      if isiterable(x):
        self.clips += x
      else:
        self.clips.append(x)
    for clip in self.clips: assert isinstance(clip, Clip), f'Got {type(clip)} instead of Clip.'
    assert len(self.clips) > 0, "Need at least one clip to form a chain."

    assert len(set(map(lambda x: x.width(), self.clips))) == 1, "Cannot chain clips because the widths do not match." + str(list(map(lambda x: x.width(), self.clips)))
    assert len(set(map(lambda x: x.height(), self.clips))) == 1, "Cannot chain clips because the heights do not match." + str(list(map(lambda x: x.heights(), self.clips)))
    assert len(set(map(lambda x: x.frame_rate(), self.clips))) == 1, "Cannot chain clips because the framerates do not match." + str(list(map(lambda x: x.frame_rate(), self.clips)))
    
    self.audio = force_audio_length(chain_audio(map(lambda x: x.get_audio(), self.clips)), self.default_audio().length())
    # TODO: Looks like chain_audio is bad choice here, especially if there are
    # many clips in the chain.  Each one many have a roundoff error in the
    # audio length, and they will all add up.  Instead, let's make a big empty
    # numpy array, and then paste in the data from each clip in the chain at
    # the right place.  That way, the errors won't accumulate they way they do
    # now, necessitating the force_audio_length above.

    # TODO: Where can we verify that each clip has the correct-length audio?
    # Maybe in realize_video()?

  def __repr__(self):
    return f'chain({self.clips})'

  def frame_rate(self):
    return self.clips[0].frame_rate()

  def width(self):
    return self.clips[0].width()

  def height(self):
    return self.clips[0].height()

  def length(self):
    return sum(map(lambda x: x.length(), self.clips))

  def find_frame_index(self, frame_index):
    # Given a frame index for the chained video, find and return the index of
    # the source clip that would contain that frame, along with the index
    # within that clip.  TODO: Binary search.
    for i in range(0, len(self.clips)):
      if frame_index < self.clips[i].length():
        return (i, frame_index)
      frame_index -= self.clips[i].length()
    assert False, f'Could not find chain element index {frame_index} within {len(self.clips)} clips, with lengths {list(map(lambda x: x.length(), self.clips))}  Total length is only {self.length()}.\n{self}'

  def frame_signature(self, index):
    i, index = self.find_frame_index(index)
    return self.clips[i].frame_signature(index)

  def get_frame(self, index):
    i, index = self.find_frame_index(index)
    return self.clips[i].get_frame(index)

  def get_audio(self):
    return self.audio


class replace_audio(Clip):
  """
  Replace the audio in a clip with something else, usually forcing the length to match.
  """

  def __init__(self, clip, audio, force_to=None):
    assert isinstance(clip, Clip)
    assert isinstance(audio, Audio)
    self.clip = clip
    self.audio = audio
    if force_to is None:
      force_to = self.frame_to_sample(self.length())
    else:
      assert isinstance(force_to, int)
    self.audio = force_audio_length(self.audio, force_to)

  def __repr__(self):
    return f'replace_audio({self.clip}, {self.audio})'

  def frame_rate(self):
    return self.clip.frame_rate()

  def width(self):
    return self.clip.width()

  def height(self):
    return self.clip.height()

  def length(self):
    return self.clip.length()

  def frame_signature(self, index):
    return self.clip.frame_signature(index)

  def get_frame(self, index):
    return self.clip.get_frame(index)

  def get_audio(self):
    return self.audio


class slice_video(Clip):
  """
  Extract the portion of a clip between the given frames.
  """
  def __init__(self, clip, start, end, units='frames'):
    sec_units = ['seconds', 'sec', 'secs', 's']
    fr_units = ['frames', 'frame', 'fr', 'f']
    sample_units = ['samples', 'sample']

    assert isinstance(clip, Clip)
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert start <= end, "Chopping %s, but end %d is before start %d." % (clip, end, start)

    self.clip = clip
    
    self.start_frame = start
    self.end_frame = end

    self.audio = clip.get_audio()
    self.audio = slice_audio(
      self.audio,
      self.frame_to_sample(self.start_frame),
      self.frame_to_sample(self.end_frame)
    )

  def __repr__(self):
    return f'slice_video({self.clip}, {self.start_frame}, {self.end_frame})'

  def frame_rate(self):
    return self.clip.frame_rate()
  def width(self):
    return self.clip.width()
  def height(self):
    return self.clip.height()
  def length(self):
    return self.end_frame - self.start_frame
  def frame_signature(self, index):
    return self.clip.frame_signature(self.start_frame + index)
  def get_frame(self, index):
    return self.clip.get_frame(self.start_frame + index)
  def get_audio(self):
    return self.audio

class superimpose(Clip):
  """Superimpose one clip on another, at a given place in each frame, starting
  at a given time."""

  def __init__(self, under_clip, over_clip, x, y, start_frame, audio='ignore'):
    assert isinstance(under_clip, Clip)
    assert isinstance(over_clip, Clip)
    assert isinstance(x, int)
    assert x >= 0
    assert isinstance(y, int)
    assert y >= 0
    assert isfloat(start_frame)
    assert start_frame >= 0
    start_frame = int(start_frame)

    assert y + over_clip.height() <= under_clip.height(), f"Superimposing {over_clip} onto {under_clip} at ({x}, {x}), but the under clip is not tall enough.  It would need to be {y+over_clip.height()}, but is only {under_clip.height()}."
    assert x + over_clip.width() <= under_clip.width(), "Superimposing %s onto %s at (%d, %d), but the under clip is not wide enough." % (over_clip.signature(), under_clip.signature(), x, y)
    assert start_frame + over_clip.length() <= under_clip.length(), f"Superimposing {over_clip} onto {under_clip} at frame {start_frame}, but the under clip is not long enough."
    assert under_clip.frame_rate() == over_clip.frame_rate(), f'Superimposing {over_clip} onto {under_clip} at frame {start_frame}, but the framerates do not match ({over_clip.frame_rate()} != {under_clip.frame_rate()}.'

    self.under_clip = under_clip
    self.over_clip = over_clip
    self.x = x
    self.y = y
    self.start_frame = start_frame

    if audio == 'ignore':
      self.audio = under_clip.get_audio()
    elif audio in ['replace', 'mix']:
      original_audio = under_clip.get_audio()
      new_audio = over_clip.get_audio()
      assert original_audio.sample_rate() == new_audio.sample_rate()
      assert original_audio.num_channels() == new_audio.num_channels()

      start_sample = under_clip.frame_to_sample(start_frame)
      end_sample = under_clip.frame_to_sample(start_frame + over_clip.length())

      before = slice_audio(original_audio, 0, start_sample)
      during = slice_audio(original_audio, start_sample, end_sample)
      after  = slice_audio(original_audio, end_sample, original_audio.length())
      
      if audio == 'replace':
        self.audio = chain_audio(before, new_audio, after)
      else: # mix
        self.audio = chain_audio(before, mix(during, new_audio), after)
    else:
      raise Exception(f'Unknown audio mode {self.audio_mode} in superimpose.')

  def __repr__(self):
    return f'superimpose({self.under_clip}, {self.over_clip}, {self.x}, {self.y}, {self.start_frame}, audio={self.audio_mode})'

  def frame_rate(self):
    return self.under_clip.frame_rate()

  def width(self):
    return self.under_clip.width()

  def height(self):
    return self.under_clip.height()

  def length(self):
    return self.under_clip.length()

  def frame_signature(self, index):
    if index >= self.start_frame and index - self.start_frame < self.over_clip.length():
      return "%s+(%s@(%d,%d,%d))" % (self.under_clip.frame_signature(index), self.over_clip.frame_signature(index - self.start_frame), self.x, self.y, self.start_frame)
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

  def get_audio(self):
    return self.audio

def superimpose_center(under_clip, over_clip, start_frame, audio='ignore'):
  """Superimpose one clip on another, in the center of each frame, starting at
  a given time."""
  assert isinstance(under_clip, Clip)
  assert isinstance(over_clip, Clip)
  assert isinstance(start_frame, int)
  x = int(under_clip.width()/2) - int(over_clip.width()/2)
  y = int(under_clip.height()/2) - int(over_clip.height()/2)
  return superimpose(under_clip, over_clip, x, y, start_frame, audio)
      

class filter_frames(Clip):
  """A clip formed by passing the frames of another clip through some function.
  The function should have one argument, the input frame, and return the output
  frame.  The output frames may have a different size from the input ones, but
  must all be the same size across the whole clip.  Audio remains unchanged."""

  def __init__(self, clip, func, name=None):
    assert isinstance(clip, Clip)
    assert callable(func)
    self.clip = clip
    self.func = func
    if name:
      self.func.__name__ = name
    self.sample_frame = func(clip.get_frame(0))
  
  def __repr__(self):
    return f'filter_frames({self.clip}, {self.func.__name__})'

  def frame_rate(self):
    return self.clip.frame_rate()

  def length(self):
    return self.clip.length()

  def width(self):
    return self.sample_frame.shape[1]

  def height(self):
    return self.sample_frame.shape[0]

  def frame_signature(self, index):
    return "%s(%s)" % (self.func.__name__, self.clip.frame_signature(index))

  def get_frame(self, index):
    return self.func(self.clip.get_frame(index))

  def get_audio(self):
    return self.clip.get_audio()

def scale_by_factor(clip, factor):
  """Scale the frames of a clip by a given factor."""
  assert isinstance(clip, Clip)
  assert isfloat(factor)
  assert factor > 0
  new_width = int(factor * clip.width())
  new_height = int(factor * clip.height())
  return scale_to_size(clip, new_width, new_height)

def scale_to_size(clip, new_width, new_height):
  """Scale the frames of a clip by a given factor, possibly distorting them."""
  assert isinstance(clip, Clip)
  assert isinstance(new_width, int), f'Width should be an integer.  Got {new_width} instead.'
  assert isinstance(new_height, int), f'Height should be an interger.  Got {new_height} instead.'
  def scale_filter(frame):
    return cv2.resize(frame, (new_width, new_height))
  scale_filter.__name__ = f'scale[{new_width}x{new_height}]'
  return filter_frames(clip, scale_filter)

class image_glob(Clip):
  """Form a video from a collection of identically-sized image files that match
  a unix-style pattern, at a given frame rate."""

  def __init__(self, pattern, frame_rate):
    self.pattern = pattern
    assert isfloat(frame_rate)
    assert frame_rate > 0

    self.frame_rate_ = frame_rate
    self.filenames = sorted(glob.glob(pattern))
    assert len(self.filenames) > 0, "No files matched pattern: " + pattern
    self.sample_frame = cv2.imread(self.filenames[0])
    assert self.sample_frame is not None

  def __repr__(self):
    return f'image_glob({self.pattern}, {self.frame_rate})'

  def frame_rate(self):
    return self.frame_rate_

  def width(self):
    return self.sample_frame.shape[1]

  def height(self):
    return self.sample_frame.shape[0]

  def length(self):
    return len(self.filenames)

  def frame_signature(self, index):
    return self.filenames[index]

  def get_frame(self, index):
    return cv2.imread(self.filenames[index])

  def get_audio(self):
    return self.default_audio()

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
  
  def __repr__(self):
    return f'repeat_frame({self.clip}, {self.frame_index}, {self.length_})'

  def frame_rate(self):
    return self.clip.frame_rate()

  def length(self):
    return self.length_

  def width(self):
    return self.clip.width()

  def height(self):
    return self.clip.height()

  def frame_signature(self, index):
    return self.clip.frame_signature(self.frame_index)

  def get_frame(self, index):
    return self.clip.get_frame(self.frame_index)

  def get_audio(self):
    return self.default_audio()

def crop(clip, lower_left, upper_right):
  """Trim the frames of a clip to show only the rectangle between lower_left and upper_right."""
  assert isinstance(clip, Clip)

  assert isinstance(lower_left[0], int)
  assert lower_left[0] >= 0

  assert isinstance(lower_left[1], int)
  assert lower_left[1] >= 0

  assert isinstance(upper_right[0], int)
  assert upper_right[0] <= clip.width()

  assert isinstance(upper_right[1], int)
  assert upper_right[1] <= clip.height()

  assert lower_left[0] < upper_right[0]
  assert lower_left[1] < upper_right[1]

  def crop_filter(frame):
    return frame[lower_left[1]:upper_right[1], lower_left[0]:upper_right[0], :]

  return filter_frames(clip, crop_filter, name=f'crop{lower_left}{upper_right}')

class force_framerate(Clip):
  """Change the frame rate at which a clip thinks it should be played.  Audio is padded or truncated to match, and thus is likely to become out-of-sync."""
  def __init__(self, clip, frame_rate):
    assert isinstance(clip, Clip)
    assert isfloat(frame_rate)
    assert frame_rate > 0

    self.clip = clip
    self.frame_rate_ = frame_rate
    self.audio = clip.get_audio()
    target_length = self.frame_to_sample(self.length())
    self.audio = force_audio_length(self.audio, target_length)
  def __repr__(self):
    return f'force_framerate({self.clip}, {self.frame_rate_})'

  def frame_rate(self): return self.frame_rate_
  def width(self): return self.clip.width()
  def height(self): return self.clip.height()
  def length(self): return self.clip.length()
  def get_frame(self, index): return self.clip.get_frame(index)
  def frame_signature(self, index): return self.clip.frame_signature(index)
  def get_audio(self): return self.audio

def fade_chain(overlap_frames, *args):
  """Concatenate several clips, with some fading overlap between them."""
  clips = list()
  for x in args:
    if isiterable(x):
      clips += x
    else:
      clips.append(x)

  # Sanity checks.
  assert isinstance(overlap_frames, int)
  assert overlap_frames >= 0

  for index, clip in enumerate(clips):
    assert isinstance(clip, Clip)
    if index == 0 or index == len(clips)-1:
      assert overlap_frames <= clip.length()
    else:
      assert 2*overlap_frames <= clip.length(), f'Clip should have been at least {2*overlap_frames} frames to chain with fading, but it only has {clip.length()} frames.'

  # For some reason, this is shown in the itertools docs, instead of being part
  # of itertools.  ?
  def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

  chunks = list()

  for ((i, a), (j, b)) in pairwise(enumerate(clips)):
    if i == 0:
      offset = 0
    else:
      offset = overlap_frames

    t1_frames = a.length()-overlap_frames-offset

    v1 = slice_video(a, offset, offset+t1_frames)
    v2 = slice_video(a, offset+t1_frames, a.length())
    v3 = slice_video(b, 0, overlap_frames)

    a2 = v2.get_audio()
    a3 = v3.get_audio()
    if a2.length() != a3.length():
      v3 = replace_audio(v3, a3, force_to=a2.length())

    chunks.append(v1)
    chunks.append(fade(v2, v3))
  v4 = slice_video(clips[-1], overlap_frames, clips[-1].length())
  chunks.append(v4)

  return chain(chunks)
      

def fade_in(clip, fade_frames, color=(0,0,0)):
  """Fade in from a solid color, defaulting to black."""
  assert isinstance(clip, Clip)
  assert isinstance(fade_frames, int)
  assert fade_frames >= 0
  assert fade_frames <= clip.length(), "Cannot fade into %s for %f frames, because the clip is only %f frames long." % (clip, fade_frames, clip.length())
  assert iscolor(color)
  blk = solid(clip.height(), clip.width(), clip.frame_rate(), fade_frames, color=color)
  return fade_chain(fade_frames, blk, clip)

def fade_out(clip, fade_frames, color=(0,0,0)):
  """Fade out to a solid color, defaulting to black."""
  assert isinstance(clip, Clip)
  assert isinstance(fade_frames, int)
  assert fade_frames >= 0
  assert fade_frames <= clip.length(), "Cannot fade out from %s for %f frames, because the clip is only %f frames long." % (clip, fade_frames, clip.length())

  blk = solid(clip.height(), clip.width(), clip.frame_rate(), fade_frames, color=color)
  return fade_chain(fade_frames, clip, blk)



class reverse(Clip):
  """Reverse the video in a clip.  No change to the audio."""
  def __init__(self, clip):
    assert isinstance(clip, Clip)
    self.clip = clip
  def __repr__(self):
    return f'reverse({self.clip})'
  def frame_rate(self):
    return self.clip.frame_rate()
  def width(self):
    return self.clip.width()
  def height(self):
    return self.clip.height()
  def length(self):
    return self.clip.length()
  def frame_signature(self, index):
    return self.clip.frame_signature(self.clip.length() - index - 1)
  def get_frame(self, index):
    return self.clip.get_frame(self.clip.length() - index - 1)
  def get_audio(self):
    return self.clip.get_audio()


class resample_frames(Clip):
  """
  Modify the frame rate and the length of the video.  No change to the audio.
  """
  def __init__(self, clip, new_frame_rate, new_length):
    assert isinstance(clip, Clip)
    assert isfloat(new_frame_rate)
    assert isinstance(new_length, int)
    self.clip = clip
    self.new_frame_rate = new_frame_rate
    self.new_length = new_length

  def new_index(self, index):
    x = int(self.clip.length()*index/self.new_length)
    return x
  def __repr__(self):
    return f'resample_frames({self.clip}, {self.new_frame_rate}, {self.new_length})'
  def frame_rate(self):
    return self.new_frame_rate
  def width(self):
    return self.clip.width()
  def height(self):
    return self.clip.height()
  def length(self):
    return self.new_length
  def frame_signature(self, index):
    return self.clip.frame_signature(self.new_index(index))
  def get_frame(self, index):
    return self.clip.get_frame(self.new_index(index))
  def get_audio(self):
    return self.clip.get_audio()

def timewarp(clip, factor):
  assert isinstance(clip, Clip)
  assert isfloat(factor)
  assert factor > 0
  a = clip.get_audio()
  x = resample_frames(clip, clip.frame_rate(), int(clip.length()*factor))
  a = timewarp_audio(a, factor)
  x = replace_audio(x, a)
  return x

def change_framerate(clip, new_frame_rate):
  assert isinstance(clip, Clip)
  assert isfloat(new_frame_rate)
  assert new_frame_rate > 0
  new_length = int(new_frame_rate*clip.length()/clip.frame_rate())
  return resample_frames(clip, new_frame_rate, int(new_frame_rate*clip.length()/clip.frame_rate()))

class pdf_page(Clip):
  """A silent video constructed from a single page of a PDF."""
  def __init__(self, pdf, page_num, length, frame_rate, **kwargs):
    assert isinstance(length, int)
    assert length > 0, f'Cannot create a PDF clip with length {length}.'
    self.pdf = pdf
    self.page_num = page_num
    self.frame_rate_ = frame_rate
    self.length_ = length
    self.hash = sha256sum(self.pdf)
    self.kwargs = kwargs
    images = pdf2image.convert_from_path(self.pdf, first_page=page_num, last_page=page_num, **kwargs)
    self.the_frame = np.array(images[0])

    # The code above seems sometimes (or always?) to give an image that has the
    # red and blue channels swapped.  Fix it.
    if images[0].mode == 'RGB':
      self.the_frame = self.the_frame[:,:,::-1]
    else:
      print("Tread carefully, because I'm not sure if a PIL image from {pdf}, which has in mode {images[0].mode} needs to have channels swapped.")

    # The code above seems sometimes to return an image that is not the
    # correct size, off by one in the width.  Fix it.
    if 'size' in kwargs:
      w = kwargs['size'][0]
      h = kwargs['size'][1]
      if h != self.the_frame.shape[0] or w != self.the_frame.shape[1]:
        self.the_frame = self.the_frame[0:h,0:w]

  def __repr__(self):
    return f'pdf_page({self.pdf}, {self.page_num}, {self.length_}, {self.frame_rate_}, {self.kwargs})'
  def frame_rate(self):
    return self.frame_rate_
  def width(self):
    return self.the_frame.shape[1]
  def height(self):
    return self.the_frame.shape[0]
  def length(self):
    return self.length_
  def get_audio(self):
    return self.default_audio()
  def frame_signature(self, index):
    return f'pdf_page: {self.pdf}, {self.hash[:5]}, {self.page_num}, {self.kwargs})'
  def get_frame(self, index):
    return self.the_frame


def loop(clip, length):
  "Repeat a clip as needed to fill the given length."
  assert isinstance(clip, Clip)
  assert isinstance(length, int)
  assert length > 0
  full_plays = int(length/clip.length())
  partial_play = length - full_plays*clip.length()
  return chain(full_plays*[clip], slice_video(clip, 0, partial_play))
  

class mono_to_stereo(Audio):
  def __init__(self, audio):
    assert isinstance(audio, Audio)
    assert audio.num_channels() == 1
    self.audio = audio

  def __repr__(self):
    return f'mono_to_stereo({self.audio})'

  def length(self):
    return self.audio.length()

  def num_channels(self):
    return 2

  def sample_rate(self):
    return self.audio.sample_rate()

  def compute_samples(self):
    data = self.audio.get_samples()
    return np.concatenate((data, data), axis=1)


if __name__ == "__main__":
  # Some basic tests/illustrations.  The source font and video are part of
  # texlive, which might be installed on your computer already.
  # TODO More here.
  # TODO Did the audio upgrade break any of these?

  font_filename = "/usr/local/texlive/2019/texmf-dist/fonts/truetype/sorkin/merriweather/MerriweatherSans-Regular.ttf"
  video_filename = "/usr/local/texlive/2019/texmf-dist/tex/latex/mwe/example-movie.mp4"

  vid = video_file(video_filename)
  vid = slice_by_frames(vid, 0, 7.5*vid.frame_rate())

  small_vid = slice_by_frames(filter_frames(vid, lambda x: cv2.resize(x, (150, 100))), 0, 100)

  vid = superimpose(vid, small_vid, 200, 100, 30)
  vid = superimpose_center(vid, small_vid, 100)

  title = add_text(black(vid.height(), vid.width(), vid.frame_rate(), 5*vid.frame_rate()), [
    (70, font_filename, "Test Video for clip.py"),
    (10, font_filename, "If you can read this, you don't need glasses.")
  ], color=(0,0,255))
  title = fade_in(title, 0.5*vid.frame_rate())
  title = fade_out(title, 0.5*vid.frame_rate())

  fade = fade_chain(title, vid, 1)

  fade.play()

