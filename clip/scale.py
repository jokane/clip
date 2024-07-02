""" Tools for scaling video. """

import cv2

from .base import require_clip
from .filter import filter_frames
from .validate import require_float, require_positive, require_int

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

