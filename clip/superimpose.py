""" A tool for pasting one clip on top of another. """

from .base import Clip, require_clip
from .validate import require_float, require_non_negative
from .composite import VideoMode, AudioMode, composite, Element

def superimpose_center(under_clip, over_clip, start_time, audio_mode=AudioMode.ADD) -> Clip:
    """Superimpose one clip on another, in the center of each frame, starting at
    a given time. |modify|"""
    require_clip(under_clip, "under clip")
    require_clip(over_clip, "over clip")
    require_float(start_time, "start time")
    require_non_negative(start_time, "start time")

    x = int(under_clip.width()/2) - int(over_clip.width()/2)
    y = int(under_clip.height()/2) - int(over_clip.height()/2)

    return composite(Element(under_clip, 0, [0,0], VideoMode.REPLACE),
                     Element(over_clip, start_time, [x,y], VideoMode.REPLACE, audio_mode))

