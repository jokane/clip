#!/usr/bin/env python3
# pylint: disable=missing-module-docstring,missing-function-docstring,wildcard-import

from pprint import pprint
import pytest
from clip3 import *

def test_is_float():
    assert is_float(0.3)
    assert is_float(3)
    assert not is_float("abc")

def test_is_color():
    assert is_color([0,0,0])
    assert not is_color([0,0,256])

def test_metrics():
    _ = Metrics(default_metrics)

    with pytest.raises(TypeError):
        Metrics(default_metrics, width=0.5)
    with pytest.raises(TypeError):
        Metrics(default_metrics, width=-1)

    Metrics(default_metrics, num_samples=0.5)

    with pytest.raises(TypeError):
        Metrics(default_metrics, num_samples=-1)

def test_solid():
    x = solid(640, 480, 30, 300, [0,0,0])
    pprint(x.frame_signature(0))
    x.get_frame(0)

def test_temporary_current_directory():
    with temporary_current_directory():
        pass

