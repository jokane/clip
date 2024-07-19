""" A tool for the "Ken Burns" effect, panning and zooming along another clip
(which is traditionally a static image, but not not be in this case.) """

# pylint: disable=duplicate-code
import math

import cv2
import numpy as np

from .base import MutatorClip
from .metrics import Metrics
from .validate import require_int_point, require_non_negative, require_less, require_less_equal
# pylint: enable=duplicate-code

class ken_burns(MutatorClip):
    """Pan and/or zoom through a clip over time. |modify|

    Crops and scales each frame of the input clip, smoothly moving the visible
    portion from a starting rectangle to an ending rectangle across the full
    duration of the input clip.

    :param clip: A clip to modify.
    :param width: The desired output width.  A positive integer.
    :param height: The desired output height.  A positive integer.
    :param start_top_left: Integer coordinates of the top-left corner of the
            visible rectangle at the start, as an `(x,y)` tuple.
    :param start_bottom_right: Integer coordinates of the bottom-right corner
            of the visible rectangle at the start, as an `(x,y)` tuple.
    :param end_top_left: Integer coordinates of the top-left corner of the
            visible rectangle at the end, as an `(x,y)` tuple.
    :param end_bottom_right: Integer coordinates of the bottom-right corner
            of the visible rectangle at the end, as an `(x,y)` tuple.

    To prevent distortion, all three of these rectangles must have the same
    aspect ratio:

        - The output clip, given by `width` and `height`.
        - The visible rectangle at the start.
        - The visible rectangle at the end.

    An exception is raised if these three are not at least approximately equal.

    """
    def __init__(self, clip, width, height, start_top_left, start_bottom_right,
                 end_top_left, end_bottom_right):
        super().__init__(clip)

        # So. Many. Ways to mess up.
        require_int_point(start_top_left, "start top left")
        require_int_point(start_bottom_right, "start bottom right")
        require_int_point(end_top_left, "end top left")
        require_int_point(end_bottom_right, "end bottom right")
        require_non_negative(start_top_left[0], "start top left x")
        require_non_negative(start_top_left[1], "start top left y")
        require_non_negative(end_top_left[0], "end top left x")
        require_non_negative(end_top_left[1], "end top left y")
        require_less(start_top_left[0], start_bottom_right[0],
                     "start top left x", "start bottom right x")
        require_less(start_top_left[1], start_bottom_right[1],
                     "start top left y", "start bottom right y")
        require_less(end_top_left[0], end_bottom_right[0],
                     "end top left x", "end bottom right x")
        require_less(end_top_left[1], end_bottom_right[1],
                     "end top left y", "end bottom right y")
        require_less_equal(start_bottom_right[0], clip.width(),
                           "start bottom right x", "clip width")
        require_less_equal(start_bottom_right[1], clip.height(),
                           "start bottom right y", "clip height")
        require_less_equal(end_bottom_right[0], clip.width(),
                           "end bottom right x", "clip width")
        require_less_equal(end_bottom_right[1], clip.height(),
                           "end bottom right y", "clip height")


        start_ratio = ((start_bottom_right[0] - start_top_left[0])
                       / (start_bottom_right[1] - start_top_left[1]))

        end_ratio = ((end_bottom_right[0] - end_top_left[0])
                     / (end_bottom_right[1] - end_top_left[1]))

        output_ratio = width/height

        if not math.isclose(start_ratio, output_ratio, abs_tol=0.1):
            raise ValueError("This ken_burns effect will distort the image at the start. "
                             f'Starting aspect ratio is {start_ratio}. '
                             f'Output aspect ratio is {output_ratio}. ')

        if not math.isclose(end_ratio, output_ratio, abs_tol=0.1):
            raise ValueError("This ken_burns effect will distort the image at the end. "
                             f'Ending aspect ratio is {end_ratio}. '
                             f'Output aspect ratio is {output_ratio}. ')

        self.start_top_left = np.array(start_top_left)
        self.start_bottom_right = np.array(start_bottom_right)
        self.end_top_left = np.array(end_top_left)
        self.end_bottom_right = np.array(end_bottom_right)

        self.metrics = Metrics(src=clip.metrics,
                               width=width,
                               height=height)

    def get_corners(self, t):
        """ Return the top left and bottom right corners of the view at the
        given frame index. """
        alpha = t/self.length()
        p1 = (((1-alpha)*self.start_top_left + alpha*self.end_top_left))
        p2 = (((1-alpha)*self.start_bottom_right + alpha*self.end_bottom_right))
        p1 = np.around(p1).astype(int)
        p2 = np.around(p2).astype(int)
        return p1, p2

    def frame_signature(self, t):
        p1, p2 = self.get_corners(t)
        return ['ken_burns', {'top_left': p1,
                              'bottom_right': p2,
                              'frame':self.clip.frame_signature(t)}]

    def get_frame(self, t):
        p1, p2 = self.get_corners(t)
        frame = self.clip.get_frame(t)
        fragment = frame[p1[1]:p2[1],p1[0]:p2[0],:]
        sized_fragment = cv2.resize(fragment, (self.width(), self.height()))
        return sized_fragment

