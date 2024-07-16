"""A function to write a clip as a reasonably-compressed, reasonably portable MP4."""

import os
import sys

from .base import require_clip
from .validate import require_string, require_float, require_positive, require_bool
from .ffmpeg import save_via_ffmpeg

def save_mp4(clip, filename, frame_rate, bitrate=None, target_size=None, two_pass=None,
         preset='slow', cache_dir='/tmp/clipcache/computed', burn_subtitles=False):
    """ Save a clip to an MP4 file.

    :param clip: The clip to save.
    :param filename: A file name to write to,
    :param frame_rate: Output frame rate in frames per second.
    :param bitrate: The target bitrate in bits per second.
    :param target_size: The target filesize in megabytes.
    :param preset: A string that controls how quickly ffmpeg encodes.  See below.
    :param two_pass: Should the `ffmpeg` encoding run twice or just once?
    :param burn_subtitles: Should the frames be modified to include the subtitle text?


    At most one of `bitrate` and `target_size` should be given.

        - If a `bitrate` is given, it will be passed along to `ffmpeg` as a
          target.
        - If a `target_size` is given, we compute the appropriate `bitrate` to
          attempt to get close to that target.
        - If both are omitted, the default is to target a bitrate of 1024k.

    For `preset`, choose from:

        - `"ultrafast"`
        - `"superfast"`
        - `"veryfast"`
        - `"faster"`
        - `"fast"`
        - `"medium"`
        - `"slow"`
        - `"slower"`
        - `"veryslow"`

    The ffmpeg documentation for these says to "use the slowest preset you have
    patience for."  Default is `slow`.

    Using `two_pass` makes things slower because the encoding process happens
    twice, but can improve the results, particularly when using `target_size`.
    Default is to use `two_pass` only when a `target_size` is given.
    """

    require_clip(clip, "clip")
    require_string(filename, "filename")
    require_float(frame_rate, "frame rate")
    require_positive(frame_rate, "frame rate")
    if bitrate is not None:
        require_float(bitrate, "bitrate")
        require_positive(bitrate, "bitrate")
    if target_size is not None:
        require_float(target_size, "target size")
        require_positive(target_size, "target size")

    # If we weren't told whether to do two passes or not, choose a sensible
    # default.
    if two_pass is None:
        two_pass = target_size is not None


    require_bool(two_pass, "two pass")
    require_string(preset, "preset")

    presets_str = "ultrafast superfast veryfast faster fast medium slow slower veryslow"
    presets = presets_str.split(' ')
    if preset not in presets:
        raise ValueError(f'Parameter preset should be one of: {presets_str}')

    # Figure out what bitrate to target.
    if bitrate is None and target_size is None:
        # A hopefully sensible high-quality default.
        bitrate = 1024*1024
    elif bitrate is None and target_size is not None:
        # Compute target bit rate, which should be in bits per second,
        # from the target filesize.
        target_bytes = 2**20 * target_size
        target_bits = 8*target_bytes
        bitrate = target_bits / clip.length()
        bitrate -= 128*1024 # Audio defaults to 1024 kilobits/second.
    elif bitrate is not None and target_size is None:
        # Nothing to do -- just use the bitrate as given.
        pass
    else:
        raise ValueError("Specify either bitrate or target_size, not both.")

    require_string(cache_dir, 'cache directory')
    require_bool(burn_subtitles, 'burn subtitles')


    # Some shared arguments across all ffmpeg calls: single pass,
    # first pass of two, and second pass of two.
    # These include filters to:
    args = []
    args.append('-vcodec libx264')
    args.append('-f mp4')
    if bitrate:
        args.append(f'-vb {bitrate}')
    if preset:
        args.append(f'-preset {preset}')
    args.append('-profile:v high')

    filters = []

    # - Set the pixel format to yuv420p, which seems to be needed
    #   to get outputs that play on Apple gadgets.
    filters.append('format=yuv420p')

    # - Ensure that the width and height are even, padding with a
    #   black row or column if needed.
    filters.append('pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2')

    # - Set the output frame rate.
    filters.append(f'fps={frame_rate}')

    # - If requested, burn in the subtitles.
    if burn_subtitles:
        filters.append('subtitles=subtitles.srt')

    filters_text=','.join(filters)
    args.append(f'-filter_complex "{filters_text}"')

    save_via_ffmpeg(clip=clip,
                    filename=filename,
                    frame_rate=frame_rate,
                    output_args=args,
                    two_pass=two_pass,
                    cache_dir=cache_dir)

def save_play_quit(clip, frame_rate, filename="spq.mp4"): # pragma: no cover
    """ Save the video, play it, and then end the process.  Useful
    sometimes when debugging, to see a particular clip without running the
    entire program. """
    clip.save(filename, frame_rate)
    os.system("mplayer " + filename)
    sys.exit(0)
