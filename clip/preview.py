"""A function to preview one or more clips in a GUI window."""

import cv2

from .base import Clip, require_clip, frame_times
from .progress import custom_progressbar
from .cache import ClipCache
from .validate import require_positive

def preview(clips, frame_rate, cache_dir='/tmp/clipcache/computed'):
    """ Render the video parts of one or more clips and display them in a
    window on screen. Can be useful to see results quickly, instead of waiting
    for things to fully render. |save|

    :param clips: A single clip or an iterable of clips to show.  Each clip
                  will be previewed in a separate window.
    :param frame_rate: Frame rate for the preview, in frames per second.
    :param cache_dir: The directory to use for the frame cache.

    """

    cache = ClipCache(cache_dir)

    if isinstance(clips, Clip):
        clips = [ clips ]

    for clip in clips:
        require_clip(clip, 'clip')

    require_positive(frame_rate, 'frame rate')
    require_positive(len(clips), 'number of clips')

    length = max(map(lambda clip: clip.length(), clips))

    if len(clips) == 1:
        label = "Previewing"
    else:
        label = f"Previewing x{len(clips)}"

    with custom_progressbar(label, length) as pb:
        pb.update(0)
        for clip in clips:
            clip.request_all_frames(frame_rate)

        for t in frame_times(length, frame_rate):
            for i, clip in enumerate(clips):
                frame = clip.get_frame_cached(cache, t)
                cv2.imshow(str(i), frame)
            pb.update(t)
            if cv2.waitKey(1) == 27: break

    for i, clip in enumerate(clips):
        cv2.destroyWindow(str(i))

