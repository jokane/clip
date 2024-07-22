""" A tool for rotating a clip around its center. """

# pylint: disable=duplicate-code
import math

import cv2
import numpy as np

from .base import MutatorClip
from .metrics import Metrics
from .validate import require_float, require_non_negative
# pylint: enable=duplicate-code

class spin(MutatorClip):
    """ Rotate the contents of a clip about its center.  |modify|

    :param clip: The clip to modify.
    :param total_rotations: A positive float indicating how many total
            rotations to make throughout the original clip's duration.

    The resulting clip will be a square, large enough to show all of the
    original clip at every point of its rotation.

    The angular velocity is computed so that the result will complete the
    requested rotations within the length of the original clip.

    """
    def __init__(self, clip, total_rotations):
        super().__init__(clip)

        require_float(total_rotations, "total rotations")
        require_non_negative(total_rotations, "total rotations")

        # Leave enough space to show the full undrlying clip at every
        # orientation.
        self.radius = math.ceil(math.sqrt(clip.width()**2 + clip.height()**2))

        self.metrics = Metrics(src=clip.metrics,
                               width=self.radius,
                               height=self.radius)

        # Figure out how much to rotate in each frame.
        rotations_per_second = total_rotations / clip.length()
        self.degrees_per_second = 360 * rotations_per_second

    def frame_signature(self, t):
        sig = self.clip.frame_signature(t)
        degrees = self.degrees_per_second * t
        return [f'rotated by {degrees}', sig]

    def get_frame(self, t):
        frame = np.zeros([self.radius, self.radius, 4], np.uint8)
        original_frame = self.clip.get_frame(t)

        a = (frame.shape[0] - original_frame.shape[0])
        b = (frame.shape[1] - original_frame.shape[1])

        frame[
            int(a/2):int(a/2)+original_frame.shape[0],
            int(b/2):int(b/2)+original_frame.shape[1],
            :
        ] = original_frame

        degrees = self.degrees_per_second * t

        # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
        image_center = tuple(np.array(frame.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, degrees, 1.0)
        rotated_frame = cv2.warpAffine(frame,
                                       rot_mat,
                                       frame.shape[1::-1],
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=[0,0,0,0])
        # Using INTER_NEAREST here instead of INTER_LINEAR, to disable
        # anti-aliasing, has two effects:
        # 1. It prevents an artifical "border" from appearing when INTER_LINEAR
        # blends "real" pixels with the background zeros around the edge of the
        # real image.  This is sort of built in if we rotate when there are
        # "real" pixels close to [0,0,0,0] background pixels.
        # 2. It gives straight lines a jagged look.
        #
        # Perhaps a better version might someday get the best of both worlds by
        # embedding the real image in a larger canvas (filled somehow with the
        # right color -- perhaps by grabbing from the boundary of the real
        # image?), rotating that larger image with INTER_LINEAR (creating an
        # ugly but distant border), and then cropping back to the radius x
        # radius size that we need.

        return rotated_frame

