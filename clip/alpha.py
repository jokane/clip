""" Tools for manupulating the alpha channel of a video. """

import numba
import numpy as np

from .base import MutatorClip
from .validate import require_callable, require_float, is_float

@numba.jit(nopython=True) # pragma: no cover
def alpha_blend(f0, f1):
    """ Blend two equally-sized RGBA images and return the result. """
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
    """ Scale the alpha channel of a given clip by the given factor, which may
    be a float (for a constant factor) or a float-returning function (for a
    factor that changes across time). |modify|"""
    def __init__(self, clip, factor):
        super().__init__(clip)

        # Make sure we got either a constant float or a callable.
        if is_float(factor):
            factor = lambda x: x
        require_callable(factor, "factor function")

        self.factor = factor

    def frame_signature(self, t):
        factor = self.factor(t)
        require_float(factor, f'factor at time {t}')
        return ['scale_alpha', self.clip.frame_signature(t), factor]

    def get_frame(self, t):
        factor = self.factor(t)
        require_float(factor, f'factor at time {t}')
        frame = self.clip.get_frame(t)
        if factor != 1.0:
            frame = frame.astype('float')
            frame[:,:,3] *= factor
            frame = frame.astype('uint8')
        return frame

