"""A function to write the audio part a clip in an audio-only format."""

import soundfile

from .base import require_clip
from .validate import require_string

def save_audio(clip, filename):
    """Save the audio part of a clip to an audio format.

    :param clip: The clip to save.
    :param filename: A file name to write the audio to.

    The file format is determined by the extension of the given filename.  The
    list of supported formats is determined by what is supported by the
    `libsndfile` library.

    """
    require_clip(clip, "clip")
    require_string(filename, "filename")
    data = clip.get_samples()
    assert data is not None
    soundfile.write(filename, data, clip.sample_rate())

