""" A tool for creating a static video from a given page of a PDF. """

import cv2
import numpy as np
import pdf2image

from .util import sha256sum_file
from .validate import require_string, require_int, require_positive, require_float
from .video import static_frame

def pdf_page(filename, page_num, length, **kwargs):
    """A silent video constructed from a single page of a PDF. |from-source|

    :param filename: The name of a PDF file.
    :param pdf_num: The page number within the PDF to extract, numbered
            starting from 1.
    :param length: The desired clip length, in seconds.
    :param kwargs: Keyword arguments to pass along to `pdf2image`.

    For `kwargs`, see the `docs for the pdf2image package
    <https://pypi.org/project/pdf2image/>`_.  Of particular interest there is
    `size=(width, height)` to get an image of a desired size.

    """
    require_string(filename, "file name")
    require_int(page_num, "page number")
    require_positive(page_num, "page number")
    require_float(length, "length")
    require_positive(length, "length")

    # Hash the file.  We'll use this in the name of the static_frame below
    # (which is used in the frame_signature there) so that things are
    # re-generated correctly when the PDF changes.
    pdf_hash = sha256sum_file(filename)

    # Get an image of the PDF.
    images = pdf2image.convert_from_path(filename,
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
                        frame_name=f'{filename} ({pdf_hash}), page {page_num} {kwargs}',
                        length=length)

