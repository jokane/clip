""" Tools for extending a video by holding a certain frame.  Useful, for
example, if you want to fade in or fade out without missing anything. """

from .base import Clip, require_clip
from .chain import chain
from .video import repeat_frame
from .validate import require_float, require_positive

def hold_at_start(clip, target_length) -> Clip:
    """Extend a clip by to fill a target length by repeating its first frame. |modify|

    :param clip: The original clip.
    :param target_length: The desired length, in seconds.

    If `target_length` is greater than `clip.length()`, the result will be
    truncated to `target_length`.

    """
    require_clip(clip, "clip")
    require_float(target_length, "target length")
    require_positive(target_length, "target length")

    # Here the repeat_frame almost certainly goes beyond target length, and
    # we force the final product to have the right length directly.  This
    # prevents getting a blank frame at the end in some cases.
    return chain(repeat_frame(clip, clip.length(), target_length),
                 clip,
                 length=target_length)

def hold_at_end(clip, target_length) -> Clip:
    """Extend a clip by to fill a target length by repeating its final frame. |modify|

    :param clip: The original clip.
    :param target_length: The desired length, in seconds.

    If `target_length` is greater than `clip.length()`, the result will be
    truncated to `target_length`.

    """
    require_clip(clip, "clip")
    require_float(target_length, "target length")
    require_positive(target_length, "target length")

    # Forcing the legnth -- same caveat as above.
    return chain(clip,
                 repeat_frame(clip, clip.length(), target_length),
                 length=target_length)

