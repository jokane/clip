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

from .audio import patch_audio_length
from .base import Clip, FiniteIndexed, require_clip, frame_times
from .from_file import parse_subtitles
from .metrics import Metrics
from .validate import require_string, require_float, require_positive, require_bool
from .progress import custom_progressbar

class from_zip(Clip, FiniteIndexed):
    """ A video clip from images stored in a zip file. |from-source|

    :param filename: The name of the zip file to read.
    :param frame_rate: The rate, in frames per second, at which the images in
            the zip file should be displayed.

    If the zip file contains a file called `audio.flac`, that file will be used
    for the audio of the resulting clip.  In this case, the video and audio
    parts must have approximately the same length.

    The resulting clip will have no subtitles.

    """

    def __init__(self, filename, frame_rate):
        Clip.__init__(self)

        require_string(filename, "file name")
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Cannot open {filename}, which does not exist or \
                    is not a file.")

        # Open the zip archive.
        self.filename = filename
        self.timestamp = os.path.getmtime(self.filename)
        self.zf = zipfile.ZipFile(filename, 'r') #pylint: disable=consider-using-with
        info_list = self.zf.infolist()

        # Find all of the images.
        image_formats = ['tga', 'jpg', 'jpeg', 'png'] # (Note: Many others could be added here.)
        pattern = ".(" + "|".join(image_formats) + ")$"

        image_info_list = filter(lambda x: re.search(pattern, x.filename), info_list)
        image_info_list = sorted(image_info_list, key=lambda x: x.filename)
        self.image_info_list = image_info_list
        FiniteIndexed.__init__(self, len(self.image_info_list), frame_rate)
        video_length = len(self.image_info_list)/frame_rate

        # Look for audio.  If it exists, we need to extract it now, instead of
        # waiting for get_samples, to determine the metrics.
        try:
            audio_info = self.zf.getinfo('audio.flac')
        except KeyError:
            audio_info = None

        if audio_info:
            audio_bytes = self.zf.read(audio_info)
            bio = io.BytesIO(audio_bytes)
            with soundfile.SoundFile(bio) as sf:
                self.samples = sf.read(always_2d=True)
                sample_rate = sf.samplerate
        else:
            sample_rate = Clip.default_metrics.sample_rate
            num_channels = Clip.default_metrics.num_channels
            self.samples = np.zeros([round(sample_rate*video_length), num_channels])

        audio_length = self.samples.shape[0]/sample_rate

        if abs(video_length - audio_length) > 1:
            raise ValueError(f'In {filename} at frame rate {frame_rate}, video and audio lengths\
                    do not match.  Video is {video_length}s; audio is {audio_length}.')

        num_samples = round(sample_rate*video_length)
        self.samples = patch_audio_length(self.samples, num_samples)


        # Figure out the metrics.
        sample_frame = self.get_frame(0)

        self.metrics = Metrics(width = sample_frame.shape[1],
                               height = sample_frame.shape[0],
                               sample_rate = sample_rate,
                               num_channels = self.samples.shape[1],
                               length = video_length)

    def frame_signature(self, t):
        index = self.time_to_frame_index(t)
        return ['zip file member', self.filename, self.timestamp,
                self.image_info_list[index].filename]

    def request_frame(self, t):
        pass

    def get_frame(self, t):
        index = self.time_to_frame_index(t)
        data = self.zf.read(self.image_info_list[index])
        pil_image = Image.open(io.BytesIO(data)).convert('RGBA')
        frame = np.array(pil_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        return frame

    def get_samples(self):
        return self.samples

    def get_subtitles(self):
        try:
            info = self.zf.getinfo('subtitles.srt')
        except KeyError:
            yield from []
            return

        srt_text = self.zf.read(info).decode('utf-8')

        yield from parse_subtitles(srt_text)


def save_zip(clip, filename, frame_rate, include_audio=True, include_subtitles=None):
    """Save a clip to a zip archive of numbered images. |save|

    :param clip: The clip to save.
    :param filename: A file name to write to.
    :param frame_rate: Output frame rate in frames per second.
    :param include_audio: Should the audio be included?
    :param include_subtitles: Should the subtitles be included? Use `None` to
            include a subtitles file only if there are more than zero subtitles
            in the clip.
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

