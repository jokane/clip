""" A tool for looping a clip over and over. """

from .base import require_clip
from .crop_slice import slice_clip
from .chain import chain
from .validate import require_float, require_positive

def loop(clip, length):
    """Repeat a clip as needed to fill the given length."""
    require_clip(clip, "clip")
    require_float(length, "length")
    require_positive(length, "length")

    full_plays = int(length/clip.length())
    partial_play = length - full_plays*clip.length()
    return chain(full_plays*[clip], slice_clip(clip, 0, partial_play))


