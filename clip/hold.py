""" Tools for extending a video by holding a certain frame.  Useful, for
example, if you want to fade in or fade out without missing anything. """

from .base import Clip, require_clip
from .chain import chain
from .video import repeat_frame
from .validate import require_float, require_positive

def hold_at_start(clip, target_length) -> Clip:
    """Extend a clip by repeating its first frame, to fill a target length.
    |modify|"""
    require_clip(clip, "clip")
    require_float(target_length, "target length")
    require_positive(target_length, "target length")

    # Here the repeat_frame almost certainly goes beyond target length, and
    # we force the final product to have the right length directly.  This
    # prevents getting a blank frame at end in some cases.
    return chain(repeat_frame(clip, clip.length(), target_length),
                 clip,
                 length=target_length)

def hold_at_end(clip, target_length) -> Clip:
    """Extend a clip by repeating its last frame, to fill a target length.
    |modify|"""
    require_clip(clip, "clip")
    require_float(target_length, "target length")
    require_positive(target_length, "target length")

    # Here the repeat_frame almost certainly goes beyond target length, and
    # we force the final product to have the right length directly.  This
    # prevents getting a blank frame at end in some cases.
    return chain(clip,
                 repeat_frame(clip, clip.length(), target_length),
                 length=target_length)

