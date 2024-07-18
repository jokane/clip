""" Tools for getting the dimensions of a video right. """

from .base import Clip, require_clip
from .scale import scale_to_fit
from .resample import resample
from .audio import stereo_to_mono, mono_to_stereo
from .composite import composite, Element, VideoMode, AudioMode
from .validate import require_int, require_positive

def letterbox(clip, width, height):
    """ Fix the clip within given dimensions, adding black bands on the
    top/bottom or left/right if needed. |modify|"""
    require_clip(clip, "clip")
    require_int(width, "width")
    require_positive(width, "width")
    require_int(height, "height")
    require_positive(height, "height")

    scaled = scale_to_fit(clip, width, height)

    position=[int((width-scaled.width())/2),
              int((height-scaled.height())/2)]

    return composite(Element(clip=scaled,
                             start_time=0,
                             position=position,
                             video_mode=VideoMode.REPLACE,
                             audio_mode=AudioMode.REPLACE),
                      width=width,
                      height=height)

def to_default_metrics(clip):
    """Adjust a clip so that its metrics match the default metrics: Scale video
    and resample to match frame rate and sample rate.  Useful if assorted clips
    from various sources will be chained together. |modify|"""

    require_clip(clip, "clip")

    dm = Clip.default_metrics

    # Video dimensions.
    if clip.width() != dm.width or clip.height() != dm.height:
        clip = letterbox(clip, dm.width, dm.height)

    # Sample rate.
    if clip.sample_rate() != dm.sample_rate:
        clip = resample(clip, sample_rate=dm.sample_rate)

    # Number of audio channels.
    nc_before = clip.num_channels()
    nc_after = dm.num_channels
    if nc_before == nc_after:
        pass
    elif nc_before == 2 and nc_after == 1:
        clip = stereo_to_mono(clip)
    elif nc_before == 1 and nc_after == 2:
        clip = mono_to_stereo(clip)
    else:
        raise NotImplementedError(f"Don't know how to convert from {nc_before}"
                                  f"channels to {nc_after}.")

    return clip
