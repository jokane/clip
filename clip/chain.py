""" For chaining together a series of clips. """

from .base import require_clip
from .fade import fade_in, fade_out
from .composite import composite, AudioMode, VideoMode, Element
from .util import flatten_args
from .validate import require_float, require_non_negative, require_equal

def chain(*args, length=None, fade_time = 0):
    """ Concatenate a series of clips.  The clips may be given individually, in
    lists or other iterables, or a mixture of both.  Optionally overlap them a
    little and fade between them."""
    # Construct our list of clips.  Flatten each list; keep each individual
    # clip.
    clips = flatten_args(args)

    # Sanity checks.
    require_float(fade_time, "fade time")
    require_non_negative(fade_time, "fade time")

    for clip in clips:
        require_clip(clip, "clip")

    if len(clips) == 0:
        raise ValueError("Need at least one clip to form a chain.")

    # Figure out when each clip should start and make a list of elements for
    # composite.
    start_time = 0
    elements = []
    for i, clip in enumerate(clips):
        if fade_time>0:
            if i>0:
                clip = fade_in(clip, fade_time)
            if i<len(clips)-1:
                clip = fade_out(clip, fade_time)
            vmode = VideoMode.ADD
        else:
            vmode = VideoMode.REPLACE

        elements.append(Element(clip=clip,
                                start_time=start_time,
                                position=(0,0),
                                video_mode=vmode,
                                audio_mode=AudioMode.ADD))

        start_time += clip.length() - fade_time

    # Let composite do all the work.
    return composite(*elements, length=length)

def fade_between(clip1, clip2):
    """ Fade from one clip to another.  Both must have the same length. """
    require_clip(clip1, "first clip")
    require_clip(clip2, "second clip")
    require_equal(clip1.length(), clip2.length(), "clip lengths")

    return chain(clip1, clip2, fade_time=clip1.length())

