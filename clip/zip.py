""" A tool for treating a zipped collection of images as a video. """

import io
import os
import re
import zipfile

import cv2
from PIL import Image
import numpy as np

from .base import Clip, VideoClip, FiniteIndexed
from .metrics import Metrics
from .validate import require_string

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

