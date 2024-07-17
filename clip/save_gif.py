"""A function to write a clip as an animated GIF."""

from .base import require_clip
from .validate import require_string, require_float, require_positive
from .ffmpeg import save_via_ffmpeg

def save_gif(clip, filename, frame_rate, cache_dir='/tmp/clipcache/computed'):
    """Save a clip to an animated GIF.

    :param clip: The clip to save.
    :param filename: A file name to write to.
    :param cache_dir: The directory to use for the frame cache."""

    require_clip(clip, "clip")
    require_string(filename, "filename")
    require_float(frame_rate, "frame rate")
    require_positive(frame_rate, "frame rate")

    args = []
    args.append('-f gif')

    filters = []
    filters.append("[0:v] split [a][b]")
    filters.append("[a] palettegen [p]")
    filters.append("[b][p] paletteuse")
    filters_text=';'.join(filters)
    args.append(f'-filter_complex "{filters_text}"')

    save_via_ffmpeg(clip=clip,
                    filename=filename,
                    frame_rate=frame_rate,
                    output_args=args,
                    two_pass=False,
                    cache_dir=cache_dir)

