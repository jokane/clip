""" Tools for manupulating the alpha channel of a video. """

import numba
import numpy as np

from .base import MutatorClip
from .validate import require_callable, require_float, is_float

@numba.jit(nopython=True) # pragma: no cover
def alpha_blend(f0, f1):
    """ Blend two equally-sized RGBA images, respecting the alpha channels of each.

    :param f0: An image.
    :param f1: Another image.
    :return: The result of alpha-blending `f0` onto `f1`.

    Note that this process, as currently implemented, is irritatingly slow,
    mostly because of the need to convert the images from `unit8` format to
    `float64` format and back.  Someday, we'll replace this with something
    better."""

    # https://stackoverflow.com/questions/28900598/how-to-combine-two-colors-with-varying-alpha-values
    # assert f0.shape == f1.shape, f'{f0.shape}!={f1.shape}'
    # assert f0.dtype == np.uint8
    # assert f1.dtype == np.uint8

    f0 = f0.astype(np.float64) / 255.0
    f1 = f1.astype(np.float64) / 255.0

    b0 = f0[:,:,0]
    g0 = f0[:,:,1]
    r0 = f0[:,:,2]
    a0 = f0[:,:,3]

    b1 = f1[:,:,0]
    g1 = f1[:,:,1]
    r1 = f1[:,:,2]
    a1 = f1[:,:,3]

    a01 = (1 - a0)*a1 + a0
    b01 = (1 - a0)*b1 + a0*b0
    g01 = (1 - a0)*g1 + a0*g0
    r01 = (1 - a0)*r1 + a0*r0

    f01 = np.zeros(shape=f0.shape, dtype=np.float64)

    f01[:,:,0] = b01
    f01[:,:,1] = g01
    f01[:,:,2] = r01
    f01[:,:,3] = a01
    f01 = (f01*255.0).astype(np.uint8)

    return f01

class scale_alpha(MutatorClip):
    """ Scale the alpha channel of a given clip by the given factor. |modify|

    :param clip: The clip to modify.
    :param factor: A positive float by which to scale the alpha channel of the
            given clip, or a callable that accepts a time and returns the
            scaling factor to use at that time.

    """
    def __init__(self, clip, factor):
        super().__init__(clip)

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

