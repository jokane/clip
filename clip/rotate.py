"""Tools to rotate a clip by multiples of 90 degrees."""

import cv2

from .filter import filter_frames

def rotate90cw(clip):
  """Rotate a clip 90 degrees counterclockwise. "|modify|"""
  return filter_frames(clip, lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE))

def rotate90ccw(clip):
  """Rotate a clip 90 degrees counterclockwise. "|modify|"""
  return filter_frames(clip, lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE))

def rotate180(clip):
    """Rotate a clip 180 degrees. |modify|"""
    return filter_frames(clip, lambda x: cv2.rotate(x, cv2.ROTATE_180))

def rotate360(clip):
    """Rotate a clip 360 degrees. |modify|"""
    return filter_frames(clip, lambda x: x)

