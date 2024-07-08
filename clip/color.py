""" Tools for manipulating the colors in a video. """

import cv2

from .base import Clip, require_clip, require_color
from .filter import filter_frames
from .composite import composite, Element, VideoMode
from .video import solid

def to_monochrome(clip) -> Clip:
    """ Convert a clip's video to monochrome. """
    def mono(frame):
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY), cv2.COLOR_GRAY2BGRA)
    return filter_frames(clip=clip,
                         func=mono,
                         name='to_monochrome',
                         size='same')

def bgr2rgb(clip) -> Clip:
    """Swap the first and third color channels.  Useful if, instead of saving,
    you are sending the frames to something, like PIL, that expects RGB instead
    of BGR."""
    def swap_channels(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)

    return filter_frames(clip=clip,
                         func=swap_channels,
                         name='bgr2rgb')

def background(clip, bg_color) -> Clip:
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
