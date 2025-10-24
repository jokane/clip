""" Tools for manupulating the alpha channel of a video. """

import numba
import numpy as np

from .base import MutatorClip, require_clip
from .validate import require_callable, require_float, is_float

@numba.njit(parallel=True)
def alpha_blend(f0, f1): #pragma: nocover
    """ Blend two equally-sized RGBA images, respecting the alpha channels of each.

    :param f0: An image.
    :param f1: Another image.
    :return: The result of alpha-blending `f0` onto `f1`.
    
    """

    h, w, _ = f0.shape
    out = np.empty_like(f0)

    for y in numba.prange(h): #pylint: disable=not-an-iterable
        for x in range(w):
            # normalized alphas in [0,1]
            a0 = f0[y, x, 3] / 255.0
            a1 = f1[y, x, 3] / 255.0

            inv_a0 = 1.0 - a0

            # blend RGB in the 0-255 domain to avoid unit mismatch
            for c in range(3):
                v = f0[y, x, c] * a0 + f1[y, x, c] * a1 * inv_a0
                # clamp to [0,255]
                if v < 0.0:
                    v = 0.0
                elif v > 255.0:
                    v = 255.0
                out[y, x, c] = np.uint8(v)

            # combined alpha: convert back to 0-255
            a_out = a0 + a1 * inv_a0
            if a_out < 0.0:
                a_out = 0.0
            elif a_out > 1.0:
                a_out = 1.0
            out[y, x, 3] = np.uint8(a_out * 255.0)

    return out

class scale_alpha(MutatorClip):
    """ Scale the alpha channel of a given clip by the given factor. |modify|

    :param clip: The clip to modify.
    :param factor: A positive float by which to scale the alpha channel of the
            given clip, or a callable that accepts a time and returns the
            scaling factor to use at that time.

    """
    def __init__(self, clip, factor):
        super().__init__(clip)

        require_clip(clip, 'clip')

        # Make sure we got either a constant float or a callable.
        if is_float(factor):
            func = lambda x: factor
        else:
            func = factor
        require_callable(func, "factor function")

        self.func = func

    def frame_signature(self, t):
        factor = self.func(t)
        require_float(factor, f'factor at time {t}')
        return ['scale_alpha', self.clip.frame_signature(t), factor]

    def get_frame(self, t):
        factor = self.func(t)
        require_float(factor, f'factor at time {t}')
        frame = self.clip.get_frame(t)
        if factor != 1.0:
            frame = frame.astype('float')
            frame[:,:,3] *= factor
            frame = frame.astype('uint8')
        return frame

