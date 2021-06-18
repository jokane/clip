#!/usr/bin/env python3
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import

import glob
import shutil

import cv2
import pytest

from clip3 import *

def test_validate():
    assert is_float(0.3)
    assert is_float(3)
    assert not is_float("abc")

    assert is_color([0,0,0])
    assert not is_color([0,0,256])

    assert not is_positive(0)
    assert is_non_negative(0)
    assert is_non_negative(5)
    assert not is_non_negative(-2)

    assert is_iterable(list())
    assert not is_iterable(123)

    require_equal(1, 1, "name")

    with pytest.raises(ValueError):
        require_equal(1, "1", "name")

def test_metrics():
    m1 = Metrics(default_metrics)
    m2 = Metrics(default_metrics, length=0.5)

    with pytest.raises(ValueError):
        m1.verify_compatible_with(m2, check_length=True)

    with pytest.raises(TypeError):
        Metrics(default_metrics, width=0.5)

    with pytest.raises(ValueError):
        Metrics(default_metrics, width=-1)

    with pytest.raises(ValueError):
        Metrics(default_metrics, length=-1)

    with pytest.raises(ValueError):
        Metrics(default_metrics, length=0)

    with pytest.raises(TypeError):
        Metrics(default_metrics, length="really long")

def test_solid():
    x = solid(640, 480, 30, 300, [0,0,0])

    x.frame_signature(0)

    img = x.get_frame(0)
    assert img.shape == (480, 640, 4)

    samples = x.get_samples()
    assert samples.shape == (x.num_samples(), x.num_channels())


def test_clip_metrics():
    secs = 30
    x = solid(640, 480, 30, secs, [0,0,0])
    assert x.length() == secs
    assert x.frame_rate() == 30
    assert x.sample_rate() == default_metrics.sample_rate
    assert x.num_samples() == secs*default_metrics.num_samples()
    assert x.num_frames() == secs*30
    assert x.num_channels() == default_metrics.num_channels
    assert f":{secs:02d}" in x.readable_length()

    x = solid(640, 480, 30, 60*60+1, [0,0,0])
    assert x.readable_length()[:2] == '1:'

def test_temporary_current_directory():
    with temporary_current_directory():
        assert glob.glob('*') == []

def test_cache():
    c = ClipCache()
    c.clear()

    x = solid(640, 480, 30, 300, [0,0,0])
    sig1 = x.frame_signature(0)
    sig2 = x.frame_signature(1)

    fname1, exists1 = c.lookup(sig1, 'png')
    fname2, exists2 = c.lookup(sig2, 'png')

    assert fname1 == fname2
    assert exists1 is False
    assert exists2 is False

    img = x.get_frame(0)
    cv2.imwrite(fname1, img)
    c.insert(fname1)

    fname1, exists1 = c.lookup(sig1, 'png')
    assert exists1 is True

    c.scan_directory()

def test_customprogressbar():
    with custom_progressbar("Testing", 100) as pb:
        pb.update(0)
        for i in range(100):
            pb.update(i)

def test_frame_to_sample():
    secs = 10
    x = solid(640, 480, 30, secs, [0,0,0])

    assert x.frame_to_sample(0) == 0
    assert x.frame_to_sample(x.frame_rate()) == x.sample_rate()

def test_ffmpeg():
    with pytest.raises(FFMPEGException):
        ffmpeg('-i /dev/zero', '/dev/null')

    with pytest.raises(FFMPEGException):
        ffmpeg('-i /dev/zero', '/dev/null', task="Testing", num_frames=100)

def test_save():
    shutil.rmtree(cache.directory)
    cache.cache = None

    x = solid(640, 480, 30, 10, [0,0,0])
    with temporary_current_directory():
        x.save('test.mp4')
        assert os.path.exists('test.mp4')

    with temporary_current_directory():
        x.save('foo.flac')
        x.save('foo.wav')
        assert os.path.exists('foo.flac')
        assert os.path.exists('foo.wav')

def test_preview():
    shutil.rmtree(cache.directory)
    cache.scan_directory()
    x = solid(640, 480, 30, 10, [0,0,0])
    x.preview()

def test_temporal_composite():
    x = solid(640, 480, 30, 5, [0,0,0])
    y = solid(640, 481, 30, 5, [255,0,0])

    with pytest.raises(ValueError):
        # Heights do not match.
        z = temporal_composite(
          (x, 0),
          (y, 6)
        )

    with pytest.raises(ValueError):
        # Can't start before 0.
        z = temporal_composite(
          (x, -1),
          (y, 6)
        )

    x = solid(640, 480, 30, 5, [0,0,0])
    y = solid(640, 480, 30, 5, [255,0,0])

    z = temporal_composite(
      (x, 0),
      (y, 6)
    )

    assert z.length() == 11

    cache.clear()
    z.preview()
    z.get_samples()

if __name__ == '__main__':  #pragma: no cover
    for name, thing in list(globals().items()):
        if 'test_' in name:
            thing()

