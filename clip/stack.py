""" Tools for stacking clips horizontally or vertically. """

from enum import Enum

from .clip import Clip
from .composite import composite, Element
from .util import flatten_args
from .validate import is_int

class Align(Enum):
    """ When stacking clips, how should each be placed? """
    CENTER = 1
    LEFT = 2
    TOP = 3
    START = 4
    RIGHT = 5
    BOTTOM = 6
    END = 7

def stack_clips(*args, align, min_dim=0, vert, name):
    """ Arrange a series of clips in a stack, either vertically or
    horizontally.  Probably use vstack or hstack to call this. """

    # Flatten things out, in case the inputs were wrapped in a list.
    clips = flatten_args(args)

    # Compute the width or height.  Do this first so we can maybe center things
    # below.
    dim = min_dim
    for clip in clips:
        if isinstance(clip, Clip):
            clip_dim = clip.width() if vert else clip.height()
            dim = max(dim, clip_dim)
        elif is_int(clip):
            pass
        else:
            raise TypeError(f"In {name}, got a {type(clip)} instead of Clip or int.")

    # Sanity check the alignment.
    if vert:
        valid_aligns = [Align.LEFT, Align.RIGHT, Align.CENTER]
    else:
        valid_aligns = [Align.TOP, Align.BOTTOM, Align.CENTER]

    if align not in valid_aligns:
        raise NotImplementedError(f"Don't know how to align {align} in {name}.")


    # Place each clip in the composite in the correct place.

    a = 0  # The coordinate that we compute each time based on align.
    b = 0  # The coordinate that moves steady forward.

    elements = []

    for clip in clips:
        if isinstance(clip, Clip):
            clip_dim = clip.width() if vert else clip.height()

            if align in [Align.LEFT, Align.TOP]:
                a = 0
            elif align==Align.CENTER:
                a = int((dim - clip_dim)/2)
            else: # align in [Align.RIGHT, Align.BOTTOM]
                a = dim - clip_dim

            elements.append(Element(clip=clip,
                                    start_time=0,
                                    position=[a, b] if vert else [b, a]))

            b += clip.height() if vert else clip.width()
        else: # must be an int, as checked above
            b += clip

    if vert:
        return composite(elements, width=dim, height=b)
    else:
        return composite(elements, height=dim, width=b)

def vstack(*args, align=Align.CENTER, min_width=0):
    """ Arrange a series of clips in a vertical stack. """
    return stack_clips(args, align=align, min_dim=min_width, vert=True, name='vstack')

def hstack(*args, align=Align.CENTER, min_height=0):
    """ Arrange a series of clips in a horizontal row. """
    return stack_clips(args, align=align, min_dim=min_height, vert=False, name='hstack')

