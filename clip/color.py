""" Tools for manipulating the colors in a video. """

import cv2

from .base import require_clip
from .filter import filter_frames
from .composite import composite, Element, AudioMode, VideoMode
from .video import solid
from .validate import check_color

def to_monochrome(clip):
    """ Convert a clip's video to monochrome. |modify|

    :param clip: A clip to modify.
    :return: A new clip, the same as the original, but with its video
            converted to monochrome.

    """

    def mono(frame):
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY), cv2.COLOR_GRAY2BGRA)

    return filter_frames(clip=clip,
                         func=mono,
                         name='to_monochrome',
                         size='same')

def bgr2rgb(clip):
    """Swap the first and third color channels.  Useful if, instead of saving,
    you are sending the frames to something, like PIL, that expects RGB instead
    of BGR. |modify|

    :param clip: A clip to modify.
    :return: A new clip, the same as the original, but with its red and blue
            swapped.

    """
    def swap_channels(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)

    return filter_frames(clip=clip,
                         func=swap_channels,
                         name='bgr2rgb')

def background(clip, bg_color):
    """ Blend a clip onto a same-sized background of the given color. |modify|

    :param clip: A clip to modify.
    :param color: A color `(r,g,b)` or `(r,g,b,a)`.  Each element must be an
            integer in the range [0,255].
    :return: A new clip, the same as the original, but with its video blended
            atop a solid background of the given `bg_color`.

    """
    require_clip(clip, 'clip')
    bg_color = check_color(bg_color, 'background color')

    return composite(Element(solid(bg_color,
                                   clip.width(),
                                   clip.height(),
                                   clip.length()),
                             0,
                             (0,0),
                             audio_mode=AudioMode.IGNORE),
                      Element(clip,
                              0,
                              (0,0),
                              video_mode=VideoMode.BLEND))
