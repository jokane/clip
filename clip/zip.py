"""Tool for treating a zipped collection of images as a video and saving clips
back to the format."""

import contextlib
import io
import os
import re
import zipfile

import cv2
from PIL import Image
import numpy as np
import soundfile

from .base import Clip, VideoClip, FiniteIndexed, require_clip, frame_times
from .metrics import Metrics
from .validate import require_string, require_float, require_positive, require_bool
from .progress import custom_progressbar

class zip_file(VideoClip, FiniteIndexed):
    """ A video clip from images stored in a zip file."""

    def __init__(self, fname, frame_rate):
        VideoClip.__init__(self)

        require_string(fname, "file name")
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"Cannot open {fname}, which does not exist or is not a file.")

        self.fname = fname
        self.zf = zipfile.ZipFile(fname, 'r') #pylint: disable=consider-using-with

        image_formats = ['tga', 'jpg', 'jpeg', 'png'] # (Note: Many others could be added here.)
        pattern = ".(" + "|".join(image_formats) + ")$"

        info_list = self.zf.infolist()
        info_list = filter(lambda x: re.search(pattern, x.filename), info_list)
        info_list = sorted(info_list, key=lambda x: x.filename)
        self.info_list = info_list
        FiniteIndexed.__init__(self, len(self.info_list), frame_rate)

        sample_frame = self.get_frame(0)

        self.metrics = Metrics(src = Clip.default_metrics,
                               width=sample_frame.shape[1],
                               height=sample_frame.shape[0],
                               length = len(self.info_list)/frame_rate)

    def frame_signature(self, t):
        index = self.time_to_frame_index(t)
        return ['zip file member', self.fname, self.info_list[index].filename]

    def request_frame(self, t):
        pass

    def get_frame(self, t):
        index = self.time_to_frame_index(t)
        data = self.zf.read(self.info_list[index])
        pil_image = Image.open(io.BytesIO(data)).convert('RGBA')
        frame = np.array(pil_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        return frame

def save_zip(clip, filename, frame_rate, include_audio=True, include_subtitles=None):
    """Save a clip to a zip archive of numbered images.

    :param clip: The clip to save.
    :param filename: A file name to write to.
    :param frame_rate: Output frame rate in frames per second.
    :param include_audio: Should the audio be included?
    :param include_subtitles: Should the subtitles be included?
    """

    require_clip(clip, "clip")
    require_string(filename, "filename")
    require_float(frame_rate, "frame rate")
    require_positive(frame_rate, "frame rate")
    require_bool(include_audio, "include audio")

    subtitles = None

    if include_subtitles is None:
        subtitles = list(clip.get_subtitles())
        include_subtitles = len(subtitles) > 0

    require_bool(include_subtitles, "include subtitles")

    if subtitles is None:
        subtitles = list(clip.get_subtitles())

    with contextlib.ExitStack() as exst:
        zf = exst.enter_context(zipfile.ZipFile(filename, 'w'))
        pb = exst.enter_context(custom_progressbar(f"Saving {filename}", round(clip.length(), 1)))

        if include_audio:
            data = clip.get_samples()
            bio = exst.enter_context(io.BytesIO())

            soundfile.write(bio, data, clip.sample_rate(), format='FLAC')
            bio.seek(0)

            with zf.open('audio.flac', 'w') as zf_member:
                zf_member.write(bio.read())

        if include_subtitles:
            sio = exst.enter_context(io.StringIO())
            clip.save_subtitles(sio)
            sio.seek(0)
            with zf.open('subtitles.srt', 'w') as zf_member:
                zf_member.write(sio.read().encode('utf-8'))

        for i, t in enumerate(frame_times(clip.length(), frame_rate)):
            pb.update(round(t, 1))
            frame = clip.get_frame(t)
            frame_compressed = cv2.imencode('.png', frame)[1]
            with zf.open(f'{i:06d}.png', 'w') as zf_member:
                zf_member.write(frame_compressed)

