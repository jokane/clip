""" Tools from combining several clips into one big composite clip. """

# pylint: disable=wildcard-import

from enum import Enum

import heapq
import numpy as np

from .alpha import alpha_blend
from .base import Clip, require_clip, frame_times
from .metrics import Metrics
from .validate import *
from .util import flatten_args


class VideoMode(Enum):
    """ When defining an element of a :class:`composite`, how should the pixels from
    this element be combined with any existing pixels that it covers, to form
    the final clip?
    
    :const VideoMode.REPLACE: Overwrite the existing pixels.

    :const VideoMode.BLEND: Use the alpha channel to blend pixels from this
          element into the existing pixels.

    :const VideoMode.ADD: Add the pixel values from this element and the
          existing pixel values.

    :const VideoMode.IGNORE: Discard the video from this element.

    Pass one of these to the constructor of :class:`Element`.
    """

    REPLACE = 1
    BLEND = 2
    ADD = 3
    IGNORE = 4

class AudioMode(Enum):
    """ When defining and element of a :class:`composite`, how should the audio
    for this element be composited into the final clip?

    :const AudioMode.REPLACE: Overwrite the existing audio.
    :const AudioMode.ADD: Add the samples from this element to the existing audio samples.
    :const AudioMode.IGNORE: Discard the audio from this element.

    Pass one of these to the constructor of :class:`Element`.
    """

    REPLACE = 5
    ADD = 6
    IGNORE = 7

class Element:
    """An element to be included in a composite."""

    def __init__(self, clip, start_time, position, video_mode=VideoMode.REPLACE,
                 audio_mode=AudioMode.REPLACE):
        require_clip(clip, "clip")
        require_float(start_time, "start_time")
        require_non_negative(start_time, "start_time")

        if is_iterable(position):
            if len(position) != 2:
                raise ValueError(f'Position should be tuple (x,y) or callable.  '
                                 f'Got {type(position)} {position} instead.')
            require_int(position[0], "position x")
            require_int(position[1], "position y")

        elif not callable(position):
            raise TypeError(f'Position should be tuple (x,y) or callable,'
                            f'not {type(position)} {position}')

        if not isinstance(video_mode, VideoMode):
            raise TypeError(f'Video mode cannot be {video_mode}.')

        if not isinstance(audio_mode, AudioMode):
            raise TypeError(f'Audio mode cannot be {audio_mode}.')

        self.clip = clip
        self.start_time = start_time
        self.position = position
        self.video_mode = video_mode
        self.audio_mode = audio_mode

    def required_dimensions(self):
        """ Return the (width, height) needed to show this element as fully as
        possible.  (May not be all of the clip, because the top left is always
        (0,0), so things at negative coordinates will still be hidden.) """
        if callable(self.position):
            nw, nh = 0, 0
            for t in frame_times(self.clip.length(), 100):
                pos = self.position(t)
                nw = max(nw, pos[0] + self.clip.width())
                nh = max(nh, pos[1] + self.clip.height())
            return (int(nw), int(nh))
        else:
            return (self.position[0] + self.clip.width(),
                    self.position[1] + self.clip.height())

    def signature(self, t):
        """ A signature for this element, to be used to create the overall
        frame signature.  Returns None if this element does not contribute at
        the given time."""
        if self.video_mode==VideoMode.IGNORE:
            return None
        if t < self.start_time or t >= self.start_time + self.clip.length():
            return None
        clip_t = t - self.start_time
        assert clip_t >= 0
        if callable(self.position):
            pos = self.position(t - self.start_time)
        else:
            pos = self.position
        return [self.video_mode, pos, self.clip.frame_signature(clip_t)]

    def get_coordinates(self, index, shape):
        """ Compute the coordinates at which this element should appear at the
        given index. """
        if callable(self.position):
            pos = self.position(index)
        else:
            pos = self.position
        x = int(pos[0])
        y = int(pos[1])
        x0 = x
        x1 = x + shape[1]
        y0 = y
        y1 = y + shape[0]
        return x0, x1, y0, y1

    def request_frame(self, t):
        """ Note that the given clip will be displayed at the given time."""
        clip_t = t - self.start_time
        if t < self.start_time or t >= self.start_time + self.clip.length():
            return
        self.clip.request_frame(clip_t)

    def get_subtitles(self):
        """ Return the subtitles of the constituent clip, shifted appropriately. """
        for subtitle_start_time, subtitle_end_time, text in self.clip.get_subtitles():
            new_start_time = self.start_time+subtitle_start_time
            new_end_time = self.start_time+subtitle_end_time
            yield new_start_time, new_end_time, text

    def apply_to_frame(self, under, t):
        """ Modify the given frame as described by this element. """
        # If this element does not apply at this index, make no change.
        clip_t = t - self.start_time
        if t < self.start_time or t >= self.start_time + self.clip.length():
            return

        # Get the frame that we're compositing in and figure out where it goes.
        over_patch = self.clip.get_frame(clip_t)
        x0, x1, y0, y1 = self.get_coordinates(clip_t, over_patch.shape)

        # If it's totally off-screen, make no change.
        if x1 < 0 or x0 > under.shape[1] or y1 < 0 or y0 > under.shape[0]:
            return

        # Clip the frame itself if needed to fit.
        if x0 < 0:
            over_patch = over_patch[:,-x0:,:]
            x0 = 0
        if x1 >= under.shape[1]:
            over_patch = over_patch[:,0:under.shape[1]-x0,:]
            x1 = under.shape[1]

        if y0 < 0:
            over_patch = over_patch[-y0:,:,:]
            y0 = 0
        if y1 >= under.shape[0]:
            over_patch = over_patch[0:under.shape[0]-y0,:,:]
            y1 = under.shape[0]

        # Actually do the compositing, based on the video mode.
        if self.video_mode == VideoMode.REPLACE:
            under[y0:y1, x0:x1, :] = over_patch
        elif self.video_mode == VideoMode.BLEND:
            under_patch = under[y0:y1, x0:x1, :]
            blended = alpha_blend(over_patch, under_patch)
            under[y0:y1, x0:x1, :] = blended
        elif self.video_mode == VideoMode.ADD:
            under[y0:y1, x0:x1, :] += over_patch
        elif self.video_mode == VideoMode.IGNORE:
            pass
        else:
            raise NotImplementedError(self.video_mode) # pragma: no cover

class composite(Clip):
    """ Given a collection of elements, form a composite clip. |modify|"""
    def __init__(self, *args, width=None, height=None, length=None):
        super().__init__()

        self.elements = flatten_args(args)

        # Sanity check on the inputs.
        for (i, e) in enumerate(self.elements):
            assert isinstance(e, Element)
            require_non_negative(e.start_time, f'start time {i}')
        if width is not None:
            require_int(width, "width")
            require_positive(width, "width")
        if height is not None:
            require_int(height, "height")
            require_positive(height, "height")
        if length is not None:
            require_float(length, "length")
            require_positive(length, "length")

        # Check for mismatches in the rates.
        e0 = self.elements[0]
        for (i, e) in enumerate(self.elements[1:]):
            require_equal(e0.clip.sample_rate(), e.clip.sample_rate(), "sample rates")

        # Compute the width, height, and length of the result.  If we're
        # given any of these, use that.  Otherwise, make it big enough for
        # every element to fit.
        if width is None or height is None:
            nw, nh = 0, 0
            for e in self.elements:
                if e.video_mode != VideoMode.IGNORE:
                    dim = e.required_dimensions()
                    nw = max(nw, dim[0])
                    nh = max(nh, dim[1])
            if width is None:
                width = nw
            if height is None:
                height = nh

        if length is None:
            length = max(map(lambda e: e.start_time + e.clip.length(), self.elements))

        self.metrics = Metrics(
          src=e0.clip.metrics,
          width=width,
          height=height,
          length=length
        )


    def frame_signature(self, t):
        sig = ['composite']
        for e in self.elements:
            esig = e.signature(t)
            if esig is not None:
                sig.append(esig)
        return sig

    def request_frame(self, t):
        for e in self.elements:
            e.request_frame(t)

    def get_frame(self, t):
        frame = np.zeros([self.metrics.height, self.metrics.width, 4], np.uint8)
        for e in self.elements:
            e.apply_to_frame(frame, t)
        return frame

    def get_samples(self):
        samples = np.zeros([self.metrics.num_samples(), self.metrics.num_channels])
        for e in self.elements:
            clip_samples = e.clip.get_samples()
            start_sample = int(e.start_time*e.clip.sample_rate())
            end_sample = start_sample + e.clip.num_samples()
            if end_sample > self.num_samples():
                end_sample = self.num_samples()
                clip_samples = clip_samples[0:self.num_samples()-start_sample]

            if e.audio_mode == AudioMode.REPLACE:
                samples[start_sample:end_sample] = clip_samples
            elif e.audio_mode == AudioMode.ADD:
                samples[start_sample:end_sample] += clip_samples
            elif e.audio_mode == AudioMode.IGNORE:
                pass
            else:
                raise NotImplementedError(e.audio_mode) # pragma: no cover

        return samples

    def get_subtitles(self):
        yield from heapq.merge(*[e.get_subtitles() for e in self.elements],
                               key = lambda x: x[0])
