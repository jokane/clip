"""A function to write a clip as an animated GIF."""

from .base import require_clip
from .validate import require_string, require_float, require_positive, require_bool
from .ffmpeg import save_via_ffmpeg

def save_gif(clip, filename, frame_rate, cache_dir='/tmp/clipcache/computed',
             burn_subtitles=False):
    """Save a clip to an animated GIF. |save|

    :param clip: The clip to save.
    :param filename: A file name to write to.
    :param cache_dir: The directory to use for the frame cache.
    :param burn_subtitles: Should the frames be modified to include the subtitle text?

    """

    require_clip(clip, "clip")
    require_string(filename, "filename")
    require_float(frame_rate, "frame rate")
    require_positive(frame_rate, "frame rate")
    require_bool(burn_subtitles, 'burn subtitles')

    args = []
    args.append('-f gif')

    filters = []
    if burn_subtitles:
        filters.append('[0:v] subtitles=subtitles.srt [c]')
    else:
        filters.append('[0:v] null [c]')

    filters.append("[c] split [a][b]")
    filters.append("[a] palettegen [p]")
    filters.append("[b][p] paletteuse [z]")
    filters_text=';'.join(filters)
    args.append(f'-filter_complex "{filters_text}"')
    args.append('-map "[z]"')

    save_via_ffmpeg(clip=clip,
                    filename=filename,
                    frame_rate=frame_rate,
                    output_args=args,
                    use_audio=False,
                    use_subtitles=True,
                    two_pass=False,
                    cache_dir=cache_dir)

