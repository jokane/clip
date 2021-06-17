#!/usr/bin/env python3
import pytest

from clip3 import *
from pprint import pprint

def test_is_float():
    assert is_float(0.3)
    assert is_float(3)
    assert not is_float("abc")

def test_is_color():
  assert is_color([0,0,0])
  assert not is_color([0,0,256])

def test_metric_copy():
    x = Metrics(default_metrics)


def test_metric_verify():
    with pytest.raises(AssertionError):
        m = Metrics(default_metrics, width=0.5)
    with pytest.raises(AssertionError):
        m = Metrics(default_metrics, width=-1)

    m = Metrics(default_metrics, num_samples=0.5)
    with pytest.raises(AssertionError):
        m = Metrics(default_metrics, num_samples=-1)

def test_clip_is_abstract():
    with pytest.raises(TypeError):
        Clip()  

def test_solid():
  x = solid(640, 480, 30, 300, [0,0,0])
  pprint(x.frame_signature(0))

test_solid()

