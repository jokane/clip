""" A tool for creating a static video from a given page of a PDF. """

import cv2
import numpy as np
import pdf2image

from .base import Clip
from .util import sha256sum_file
from .validate import require_string, require_int, require_positive, require_float
from .video import static_frame

def pdf_page(pdf_file, page_num, length, **kwargs) -> Clip:
    """A silent video constructed from a single page of a PDF."""
    require_string(pdf_file, "file name")
    require_int(page_num, "page number")
    require_positive(page_num, "page number")
    require_float(length, "length")
    require_positive(length, "length")

    # Hash the file.  We'll use this in the name of the static_frame below
    # (which is used in the frame_signature there) so that things are
    # re-generated correctly when the PDF changes.
    pdf_hash = sha256sum_file(pdf_file)

    # Get an image of the PDF.
    images = pdf2image.convert_from_path(pdf_file,
                                         first_page=page_num,
                                         last_page=page_num,
                                         **kwargs)
    image = images[0].convert('RGBA')
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)

    # Sometimes we get, for reasons not adequately understood, an image that is
    # not the correct size, off by one in the width.  Fix it.
    if 'size' in kwargs:
        w = kwargs['size'][0]
        h = kwargs['size'][1]
        if h != frame.shape[0] or w != frame.shape[1]:
            frame = frame[0:h,0:w]  # pragma: no cover

    # Form a clip that shows this image repeatedly.
    return static_frame(frame,
                        frame_name=f'{pdf_file} ({pdf_hash}), page {page_num} {kwargs}',
                        length=length)

