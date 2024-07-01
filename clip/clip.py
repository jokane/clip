""" A monolith for now. """

# pylint: disable=too-many-lines
# pylint: disable=wildcard-import

import glob
import heapq
import hashlib
import io
import math
import os
import re
import zipfile

import cv2
import numpy as np
import pdf2image
from PIL import Image

from .audio import stereo_to_mono, mono_to_stereo
from .base import Clip, VideoClip, AudioClip, MutatorClip, FiniteIndexed, require_clip
from .chain import chain
from .composite import composite, Element, AudioMode, VideoMode
from .ffmpeg import *
from .filter import filter_frames
from .metrics import *
from .progress import custom_progressbar
from .resample import resample
from .util import *
from .validate import *


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
    request_frame = AudioClip.request_frame
    get_frame = AudioClip.get_frame
    get_samples = VideoClip.get_samples
    get_subtitles = VideoClip.get_subtitles

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

    def get_subtitles(self):
        return []

def black(width, height, length):
    """ A silent solid black clip. """
    return solid([0,0,0], width, height, length)

def white(width, height, length):
    """ A silent white black clip. """
    return solid([255,255,255], width, height, length)

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

        self.start_time = start
        self.end_time = end
        self.start_sample = int(start * self.sample_rate())
        self.metrics = Metrics(self.metrics, length=end-start)

        self.subtitles = None

    def frame_signature(self, t):
        return self.clip.frame_signature(self.start_time + t)

    def request_frame(self, t):
        self.clip.request_frame(self.start_time + t)

    def get_frame(self, t):
        return self.clip.get_frame(self.start_time + t)

    def get_samples(self):
        original_samples = self.clip.get_samples()
        return original_samples[self.start_sample:self.start_sample+self.num_samples()]

    def get_subtitles(self):
        if self.subtitles is None:
            self.subtitles = []
            for subtitle in self.clip.get_subtitles():
                new_start = subtitle[0] - self.start_time
                new_end = subtitle[1] - self.start_time
                length = self.length()
                if 0 <= new_start <= length or 0 <= new_end <= length:
                    new_start = max(0, new_start)
                    new_end = min(self.length(), new_end)
                    self.subtitles.append((new_start, new_end, subtitle[2]))
        return self.subtitles

class static_frame(VideoClip):
    """ Show a single image over and over, silently. """
    def __init__(self, the_frame, frame_name, length):
        super().__init__()
        try:
            height, width, depth = the_frame.shape
        except AttributeError as e:
            raise TypeError(f"Cannot not get shape of {the_frame}.") from e
        except ValueError as e:
            raise ValueError(f"Could not get width, height, and depth of {the_frame}."
              f" Shape is {the_frame.shape}.") from e
        if depth != 4:
            raise ValueError(f"Frame {the_frame} does not have 4 channels."
              f" Shape is {the_frame.shape}.")

        self.metrics = Metrics(src=Clip.default_metrics,
                               width=width,
                               height=height,
                               length=length)

        self.the_frame = the_frame.copy()
        hash_source = self.the_frame.tobytes()
        self.sig = hashlib.sha1(hash_source).hexdigest()[:7]

        self.frame_name = frame_name

    def frame_signature(self, t):
        return [ 'static_frame', {
          'name': self.frame_name,
          'sig': self.sig
        }]

    def request_frame(self, t):
        pass

    def get_frame(self, t):
        return self.the_frame

    def get_subtitles(self):
        return []

def static_image(filename, length):
    """ Show a single image loaded from a file over and over, silently. """
    the_frame = read_image(filename)
    assert the_frame is not None
    return static_frame(the_frame, filename, length)

def scale_by_factor(clip, factor):
    """Scale the frames of a clip by a given factor."""
    require_clip(clip, "clip")
    require_float(factor, "scaling factor")
    require_positive(factor, "scaling factor")

    new_width = int(factor * clip.width())
    new_height = int(factor * clip.height())
    return scale_to_size(clip, new_width, new_height)

def scale_to_fit(clip, max_width, max_height):
    """Scale the frames of a clip to fit within the given constraints,
    maintaining the aspect ratio."""

    aspect1 = clip.width() / clip.height()
    aspect2 = max_width / max_height

    if aspect1 > aspect2:
        # Fill width.
        new_width = max_width
        new_height = clip.height() * max_width / clip.width()
    else:
        # Fill height.
        new_height = max_height
        new_width = clip.width() * max_height / clip.height()

    return scale_to_size(clip, int(new_width), int(new_height))

def scale_to_size(clip, width, height):
    """Scale the frames of a clip to a given size, possibly distorting them."""
    require_clip(clip, "clip")
    require_int(width, "new width")
    require_positive(width, "new width")
    require_int(height, "new height")
    require_positive(height, "new height")

    def scale_filter(frame):
        return cv2.resize(frame, (width, height), cv2.INTER_CUBIC)

    return filter_frames(clip=clip,
                         func=scale_filter,
                         name=f'scale to {width}x{height}',
                         size=(width,height))

class reverse(MutatorClip):
    """ Reverse both the video and audio in a clip. """
    def frame_signature(self, t):
        return self.clip.frame_signature(self.length() - t)
    def get_frame(self, t):
        return self.clip.get_frame(self.length() - t)
    def get_samples(self):
        return np.flip(self.clip.get_samples(), axis=0)

class crop(MutatorClip):
    """Trim the frames of a clip to show only the rectangle between
    lower_left and upper_right."""
    def __init__(self, clip, lower_left, upper_right):
        super().__init__(clip)
        require_int(lower_left[0], "lower left")
        require_non_negative(lower_left[0], "lower left")
        require_int(lower_left[1], "lower left")
        require_non_negative(lower_left[1], "lower left")
        require_int(upper_right[0], "upper right")
        require_less_equal(upper_right[0], clip.width(), "upper right", "width")
        require_int(upper_right[1], "upper right")
        require_less_equal(upper_right[1], clip.height(), "upper right", "width")
        require_less(lower_left[0], upper_right[0], "lower left", "upper right")
        require_less(lower_left[1], upper_right[1], "lower left", "upper right")

        self.lower_left = lower_left
        self.upper_right = upper_right

        self.metrics = Metrics(self.metrics,
                               width=upper_right[0]-lower_left[0],
                               height=upper_right[1]-lower_left[1])

    def frame_signature(self, t):
        return ['crop', self.lower_left, self.upper_right, self.clip.frame_signature(t)]

    def get_frame(self, t):
        frame = self.clip.get_frame(t)
        ll = self.lower_left
        ur = self.upper_right
        return frame[ll[1]:ur[1], ll[0]:ur[0], :]


def to_monochrome(clip):
    """ Convert a clip's video to monochrome. """
    def mono(frame):
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY), cv2.COLOR_GRAY2BGRA)
    return filter_frames(clip=clip,
                         func=mono,
                         name='to_monochrome',
                         size='same')

def slice_out(clip, start, end):
    """ Remove the part between the given endponts. """
    require_clip(clip, "clip")
    require_float(start, "start time")
    require_non_negative(start, "start time")
    require_float(end, "end time")
    require_positive(end, "end time")
    require_less(start, end, "start time", "end time")
    require_less_equal(end, clip.length(), "end time", "clip length")

    # Special cases because slice_clip complains if we ask for a length zero
    # slice.  As a bonus, this eliminates the need for chaining. (And, if for
    # some reason, we're asked to slice out the entire clip, the aforementioned
    # slice_clip complaint will kick in.)
    if start == 0:
        return slice_clip(clip, end, clip.length())
    elif end == clip.length():
        return slice_clip(clip, 0, start)

    # Normal case: Something before and something after.
    return chain(slice_clip(clip, 0, start),
                 slice_clip(clip, end, clip.length()))


def letterbox(clip, width, height):
    """ Fix the clip within given dimensions, adding black bands on the
    top/bottom or left/right if needed. """
    require_clip(clip, "clip")
    require_int(width, "width")
    require_positive(width, "width")
    require_int(height, "height")
    require_positive(height, "height")

    scaled = scale_to_fit(clip, width, height)

    position=[int((width-scaled.width())/2),
              int((height-scaled.height())/2)]

    return composite(Element(clip=scaled,
                             start_time=0,
                             position=position,
                             video_mode=VideoMode.REPLACE,
                             audio_mode=AudioMode.REPLACE),
                      width=width,
                      height=height)

class repeat_frame(VideoClip):
    """Show the same frame, from another clip, over and over."""

    def __init__(self, clip, when, length):
        super().__init__()
        require_clip(clip, "clip")
        require_float(when, "time")
        require_non_negative(when, "time")
        require_less_equal(when, clip.length(), "time", "clip length")
        require_float(length, "length")
        require_positive(length, "length")

        self.metrics = Metrics(src=clip.metrics,
                               length=length)
        self.clip = clip
        self.when = when

    def frame_signature(self, t):
        return self.clip.frame_signature(self.when)

    def request_frame(self, t):
        self.clip.request_frame(self.when)

    def get_frame(self, t):
        return self.clip.get_frame(self.when)

    def get_subtitles(self):
        return []

def hold_at_start(clip, target_length):
    """Extend a clip by repeating its first frame, to fill a target length."""
    require_clip(clip, "clip")
    require_float(target_length, "target length")
    require_positive(target_length, "target length")

    # Here the repeat_frame almost certainly goes beyond target length, and
    # we force the final product to have the right length directly.  This
    # prevents getting a blank frame at end in some cases.
    return chain(repeat_frame(clip, clip.length(), target_length),
                 clip,
                 length=target_length)

def hold_at_end(clip, target_length):
    """Extend a clip by repeating its last frame, to fill a target length."""
    require_clip(clip, "clip")
    require_float(target_length, "target length")
    require_positive(target_length, "target length")

    # Here the repeat_frame almost certainly goes beyond target length, and
    # we force the final product to have the right length directly.  This
    # prevents getting a blank frame at end in some cases.
    return chain(clip,
                 repeat_frame(clip, clip.length(), target_length),
                 length=target_length)

class image_glob(VideoClip,FiniteIndexed):
    """Video from a collection of identically-sized image files that match a
    unix-style pattern, at a given frame rate or timed to a given length."""
    def __init__(self, pattern, frame_rate=None, length=None):
        VideoClip.__init__(self)

        require_string(pattern, "pattern")

        self.pattern = pattern

        self.filenames = sorted(glob.glob(pattern))
        if len(self.filenames) == 0:
            raise FileNotFoundError(f'No files matched pattern: {pattern}')

        # Get full pathnames, in case the current directory changes.
        self.filenames = list(map(lambda x: os.path.join(os.getcwd(), x), self.filenames))
        FiniteIndexed.__init__(self, len(self.filenames), frame_rate, length)

        sample_frame = cv2.imread(self.filenames[0])
        assert sample_frame is not None

        self.metrics = Metrics(src=Clip.default_metrics,
                               width=sample_frame.shape[1],
                               height=sample_frame.shape[0],
                               length = len(self.filenames)/self.frame_rate)

    def frame_signature(self, t):
        return self.filenames[self.time_to_frame_index(t)]

    def request_frame(self, t):
        pass

    def get_frame(self, t):
        return read_image(self.filenames[self.time_to_frame_index(t)])

class zip_file(VideoClip, FiniteIndexed):
    """ A video clip from images stored in a zip file."""

    def __init__(self, fname, frame_rate):
        VideoClip.__init__(self)

        require_string(fname, "file name")
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Cannot open {fname}, which does not exist or is not a file.")

        self.fname = fname
        self.zf = zipfile.ZipFile(fname, 'r') #pylint: disable=consider-using-with

        image_formats = ['tga', 'jpg', 'jpeg', 'png'] # (Note: Many others could be added here.)
        pattern = ".(" + "|".join(image_formats) + ")$"

        info_list = self.zf.infolist()
        info_list = filter(lambda x: re.search(pattern, x.filename), info_list)
        info_list = sorted(info_list, key=lambda x: x.filename)
        self.info_list = info_list
        FiniteIndexed.__init__(self, len(self.info_list), frame_rate)

        sample_frame = self.get_frame(0)

        self.metrics = Metrics(src = Clip.default_metrics,
                               width=sample_frame.shape[1],
                               height=sample_frame.shape[0],
                               length = len(self.info_list)/frame_rate)

    def frame_signature(self, t):
        index = self.time_to_frame_index(t)
        return ['zip file member', self.fname, self.info_list[index].filename]

    def request_frame(self, t):
        pass

    def get_frame(self, t):
        index = self.time_to_frame_index(t)
        data = self.zf.read(self.info_list[index])
        pil_image = Image.open(io.BytesIO(data)).convert('RGBA')
        frame = np.array(pil_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        return frame

def pdf_page(pdf_file, page_num, length, **kwargs):
    """A silent video constructed from a single page of a PDF."""
    require_string(pdf_file, "file name")
    require_int(page_num, "page number")
    require_positive(page_num, "page number")
    require_float(length, "length")
    require_positive(length, "length")

    # Hash the file.  We'll use this in the name of the static_frame below
    # (which is used in the frame_signature there) so that things are
    # re-generated correctly when the PDF changes.
    pdf_hash = sha256sum_file(pdf_file)

    # Get an image of the PDF.
    images = pdf2image.convert_from_path(pdf_file,
                                         first_page=page_num,
                                         last_page=page_num,
                                         **kwargs)
    image = images[0].convert('RGBA')
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)

    # Sometimes we get, for reasons not adequately understood, an image that is
    # not the correct size, off by one in the width.  Fix it.
    if 'size' in kwargs:
        w = kwargs['size'][0]
        h = kwargs['size'][1]
        if h != frame.shape[0] or w != frame.shape[1]:
            frame = frame[0:h,0:w]  # pragma: no cover

    # Form a clip that shows this image repeatedly.
    return static_frame(frame,
                        frame_name=f'{pdf_file} ({pdf_hash}), page {page_num} {kwargs}',
                        length=length)

class spin(MutatorClip):
    """ Rotate the contents of a clip about the center, a given number of
    times. Rotational velocity is computed to complete the requested rotations
    within the length of the original clip."""
    def __init__(self, clip, total_rotations):
        super().__init__(clip)

        require_float(total_rotations, "total rotations")
        require_non_negative(total_rotations, "total rotations")

        # Leave enough space to show the full undrlying clip at every
        # orientation.
        self.radius = math.ceil(math.sqrt(clip.width()**2 + clip.height()**2))

        self.metrics = Metrics(src=clip.metrics,
                               width=self.radius,
                               height=self.radius)

        # Figure out how much to rotate in each frame.
        rotations_per_second = total_rotations / clip.length()
        self.degrees_per_second = 360 * rotations_per_second

    def frame_signature(self, t):
        sig = self.clip.frame_signature(t)
        degrees = self.degrees_per_second * t
        return [f'rotated by {degrees}', sig]

    def get_frame(self, t):
        frame = np.zeros([self.radius, self.radius, 4], np.uint8)
        original_frame = self.clip.get_frame(t)

        a = (frame.shape[0] - original_frame.shape[0])
        b = (frame.shape[1] - original_frame.shape[1])

        frame[
            int(a/2):int(a/2)+original_frame.shape[0],
            int(b/2):int(b/2)+original_frame.shape[1],
            :
        ] = original_frame

        degrees = self.degrees_per_second * t

        # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
        image_center = tuple(np.array(frame.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, degrees, 1.0)
        rotated_frame = cv2.warpAffine(frame,
                                       rot_mat,
                                       frame.shape[1::-1],
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=[0,0,0,0])
        # Using INTER_NEAREST here instead of INTER_LINEAR, to disable
        # anti-aliasing, has two effects:
        # 1. It prevents an artifical "border" from appearing when INTER_LINEAR
        # blends "real" pixels with the background zeros around the edge of the
        # real image.  This is sort of built in if we rotate when there are
        # "real" pixels close to [0,0,0,0] background pixels.
        # 2. It gives straight lines a jagged look.
        #
        # Perhaps a better version might someday get the best of both worlds by
        # embedding the real image in a larger canvas (filled somehow with the
        # right color -- perhaps by grabbing from the boundary of the real
        # image?), rotating that larger image with INTER_LINEAR (creating an
        # ugly but distant border), and then cropping back to the radius x
        # radius size that we need.

        return rotated_frame

def background(clip, bg_color):
    """ Blend a clip onto a same-sized background of the given color. """
    require_clip(clip, 'clip')
    require_color(bg_color, 'background color')

    return composite(Element(solid(bg_color,
                                   clip.width(),
                                   clip.height(),
                                   clip.length()),
                             0,
                             (0,0)),
                      Element(clip,
                              0,
                              (0,0),
                              video_mode=VideoMode.BLEND))

def superimpose_center(under_clip, over_clip, start_time, audio_mode=AudioMode.ADD):
    """Superimpose one clip on another, in the center of each frame, starting at
    a given time."""
    require_clip(under_clip, "under clip")
    require_clip(over_clip, "over clip")
    require_float(start_time, "start time")
    require_non_negative(start_time, "start time")

    x = int(under_clip.width()/2) - int(over_clip.width()/2)
    y = int(under_clip.height()/2) - int(over_clip.height()/2)

    return composite(Element(under_clip, 0, [0,0], VideoMode.REPLACE),
                     Element(over_clip, start_time, [x,y], VideoMode.REPLACE, audio_mode))

def loop(clip, length):
    """Repeat a clip as needed to fill the given length."""
    require_clip(clip, "clip")
    require_float(length, "length")
    require_positive(length, "length")

    full_plays = int(length/clip.length())
    partial_play = length - full_plays*clip.length()
    return chain(full_plays*[clip], slice_clip(clip, 0, partial_play))

class ken_burns(MutatorClip):
    """Pan and/or zoom through a clip over time."""
    def __init__(self, clip, width, height, start_top_left, start_bottom_right,
                 end_top_left, end_bottom_right):
        super().__init__(clip)

        # So. Many. Ways to mess up.
        require_int_point(start_top_left, "start top left")
        require_int_point(start_bottom_right, "start bottom right")
        require_int_point(end_top_left, "end top left")
        require_int_point(end_bottom_right, "end bottom right")
        require_non_negative(start_top_left[0], "start top left x")
        require_non_negative(start_top_left[1], "start top left y")
        require_non_negative(end_top_left[0], "end top left x")
        require_non_negative(end_top_left[1], "end top left y")
        require_less(start_top_left[0], start_bottom_right[0],
                     "start top left x", "start bottom right x")
        require_less(start_top_left[1], start_bottom_right[1],
                     "start top left y", "start bottom right y")
        require_less(end_top_left[0], end_bottom_right[0],
                     "end top left x", "end bottom right x")
        require_less(end_top_left[1], end_bottom_right[1],
                     "end top left y", "end bottom right y")
        require_less_equal(start_bottom_right[0], clip.width(),
                           "start bottom right x", "clip width")
        require_less_equal(start_bottom_right[1], clip.height(),
                           "start bottom right y", "clip height")
        require_less_equal(end_bottom_right[0], clip.width(),
                           "end bottom right x", "clip width")
        require_less_equal(end_bottom_right[1], clip.height(),
                           "end bottom right y", "clip height")


        start_ratio = ((start_bottom_right[0] - start_top_left[0])
                       / (start_bottom_right[1] - start_top_left[1]))

        end_ratio = ((end_bottom_right[0] - end_top_left[0])
                     / (end_bottom_right[1] - end_top_left[1]))

        output_ratio = width/height

        if not math.isclose(start_ratio, output_ratio, abs_tol=0.1):
            raise ValueError("This ken_burns effect will distort the image at the start. "
                             f'Starting aspect ratio is {start_ratio}. '
                             f'Output aspect ratio is {output_ratio}. ')

        if not math.isclose(end_ratio, output_ratio, abs_tol=0.1):
            raise ValueError("This ken_burns effect will distort the image at the end. "
                             f'Ending aspect ratio is {end_ratio}. '
                             f'Output aspect ratio is {output_ratio}. ')

        self.start_top_left = np.array(start_top_left)
        self.start_bottom_right = np.array(start_bottom_right)
        self.end_top_left = np.array(end_top_left)
        self.end_bottom_right = np.array(end_bottom_right)

        self.metrics = Metrics(src=clip.metrics,
                               width=width,
                               height=height)

    def get_corners(self, t):
        """ Return the top left and bottom right corners of the view at the
        given frame index. """
        alpha = t/self.length()
        p1 = (((1-alpha)*self.start_top_left + alpha*self.end_top_left))
        p2 = (((1-alpha)*self.start_bottom_right + alpha*self.end_bottom_right))
        p1 = np.around(p1).astype(int)
        p2 = np.around(p2).astype(int)
        return p1, p2

    def frame_signature(self, t):
        p1, p2 = self.get_corners(t)
        return ['ken_burns', {'top_left': p1,
                              'bottom_right': p2,
                              'frame':self.clip.frame_signature(t)}]

    def get_frame(self, t):
        p1, p2 = self.get_corners(t)
        frame = self.clip.get_frame(t)
        fragment = frame[p1[1]:p2[1],p1[0]:p2[0],:]
        sized_fragment = cv2.resize(fragment, (self.width(), self.height()))
        return sized_fragment

class add_subtitles(MutatorClip):
    """ Add one or more subtitles to a clip. """
    def __init__(self, clip, *args):
        super().__init__(clip)
        for i, subtitle in enumerate(args):
            require_iterable(subtitle, f'subtitle {i}')
            require_float(subtitle[0], f'subtitle {i} start time')
            require_non_negative(subtitle[0], f'subtitle {i} start time')
            require_less(subtitle[0], subtitle[1], f'subtitle {i} start time',
                         f'subtitle {i} end time')
            require_less_equal(subtitle[1], clip.length(), f'subtitle {i} start time',
                               'clip length')
            require_string(subtitle[2], f'subtitle {i} text')

        self.new_subtitles = args
        self.subtitles = None

    def get_subtitles(self):
        if self.subtitles is None:
            self.subtitles = list(heapq.merge(self.new_subtitles, self.clip.get_subtitles()))
        yield from self.subtitles

def bgr2rgb(clip):
    """Swap the first and third color channels.  Useful if, instead of saving,
    you are sending the frames to something, like PIL, that expects RGB instead
    of BGR."""
    def swap_channels(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)

    return filter_frames(clip=clip,
                         func=swap_channels,
                         name='bgr2rgb')

def to_default_metrics(clip):
    """Adjust a clip so that its metrics match the default metrics: Scale video
    and resample to match frame rate and sample rate.  Useful if assorted clips
    from various sources will be chained together."""

    require_clip(clip, "clip")

    dm = Clip.default_metrics

    # Video dimensions.
    if clip.width() != dm.width or clip.height() != dm.height:
        clip = letterbox(clip, dm.width, dm.height)

    # Sample rate.
    if clip.sample_rate() != dm.sample_rate:
        clip = resample(clip, sample_rate=dm.sample_rate)

    # Number of audio channels.
    nc_before = clip.num_channels()
    nc_after = dm.num_channels
    if nc_before == nc_after:
        pass
    elif nc_before == 2 and nc_after == 1:
        clip = stereo_to_mono(clip)
    elif nc_before == 1 and nc_after == 2:
        clip = mono_to_stereo(clip)
    else:
        raise NotImplementedError(f"Don't know how to convert from {nc_before}"
                                  f"channels to {nc_after}.")

    return clip
