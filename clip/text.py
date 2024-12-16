""" A tool for creating clips that display text. """
import os

import numpy as np
from PIL import ImageFont, ImageDraw, Image

from .alpha import alpha_blend
from .base import Clip, VideoClip
from .metrics import Metrics
from .validate import (require_string, require_float, require_positive,
                       check_color, require_non_negative)


def get_font(font, size):
    """Return a TrueType font for use on `Pillow` images.

    :param font: The filename of the desired font.
    :param size: The desired size, in pixels.

    :return: A `Pillow` ImageFont.

    This differs from calling `ImageFont.truetype()` directly only by caching
    to prevent loading the same font again and again. The performance
    improvement from this caching seems to be small but non-zero.

    """
    if (font, size) not in get_font.cache:
        try:
            get_font.cache[(font, size)] = ImageFont.truetype(font, size)
        except OSError as e:
            raise ValueError(f"Failed to open font {font}.") from e
    return get_font.cache[(font, size)]
get_font.cache = {}

class draw_text(VideoClip):
    """ A clip consisting of just a bit of text.  |ex-nihilo|

    :param text: The string of text to draw.
    :param font_filename: The filename of a TrueType font.
    :param color: A color `(r,g,b)` or `(r,g,b,a)`.  Each element must be an
            integer in the range [0,255].
    :param size: The desired size, in pixels.
    :param length: The length of the clip in seconds.  A positive float.
    :param outline_width: The size of the desired outline, in pixels.
    :param outline_color: The color of the desired outline, given as `(r,g,b)`
            or `(r,g,b,a)`.  Each element must be an integer in the range [0,255].

    The resulting clip will be the right size to contain the desired text,
    which will be draw in the given color on a transparent background.

    If either of `outline_width` or `outline_color` are given, both must be
    given.

    """
    def __init__(self, text, font_filename, font_size, color, length,
                 outline_width=None, outline_color=None):
        super().__init__()

        require_string(font_filename, "font filename")
        require_float(font_size, "font size")
        require_positive(font_size, "font size")
        color = check_color(color, "color")

        if (outline_width is None) ^ (outline_color is None):
            raise ValueError('Got only one of outline width and outline color. '
                             'Should have been neither or both.')

        if outline_width is not None:
            require_float(outline_width, "outline width")
            require_non_negative(outline_width, "outline width")
        else:
            outline_width = 0

        if outline_color is not None:
            outline_color = check_color(outline_color, "outline color")

        # Determine the bounding box for the text that we want.  This is
        # relevant both for sizing and for using the right position later when
        # we actually draw the text.
        draw = ImageDraw.Draw(Image.new("RGBA", (1,1)))
        font = get_font(font_filename, font_size)

        self.bbox = draw.textbbox((0,0), text=text, font=font, stroke_width=outline_width)

        self.metrics = Metrics(src=Clip.default_metrics,
                               width=self.bbox[2]-self.bbox[0],
                               height=self.bbox[3]-self.bbox[1],
                               length=length)

        self.text = text
        self.font_filename = font_filename
        self.font_timestamp = os.path.getmtime(self.font_filename)
        self.font_size = font_size
        self.color = color
        self.outline_width = outline_width
        self.outline_color = outline_color
        self.frame = None

    def frame_signature(self, t):
        return ['text', self.text, self.font_filename, self.font_timestamp,
                self.font_size, self.color, self.outline_width, self.outline_color]

    def request_frame(self, t):
        pass

    def get_frame(self, t):
        if self.frame is None:
            # Use Pillow to draw the text.
            image = Image.new("RGBA", (self.width(), self.height()), (0,0,0,0))
            draw = ImageDraw.Draw(image)
            color = self.color
            ocolor = self.outline_color
            if ocolor is None:
                draw.text((-self.bbox[0],-self.bbox[1]),
                          self.text,
                          font=get_font(self.font_filename, self.font_size),
                          fill=(color[2],color[1],color[0],255))
            else:
                draw.text((-self.bbox[0],-self.bbox[1]),
                          self.text,
                          font=get_font(self.font_filename, self.font_size),
                          fill=(color[2],color[1],color[0],255),
                          stroke_width=self.outline_width,
                          stroke_fill=(ocolor[2],ocolor[1],ocolor[0],255))

            frame = np.array(image)

            # Pillow seems not to handle transparency quite how one might
            # expect -- as far as I can tell, it seems to fill the entire text
            # rectangle with the target color, and then use the alpha channel
            # to "draw" the text.  These seemed to be resulting in rectangular
            # blobs, instead of readable text in some cases.  (Hypothesis:
            # Sometimes the alpha channel is discarded at some point?)  Below,
            # we fix this by blending into a black background.
            bg = np.zeros(frame.shape, dtype=np.uint8)
            frame = alpha_blend(frame, bg)
            self.frame = frame

        return self.frame
