#!/usr/bin/env python3
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import
# pylint: disable=too-many-lines

import io
import urllib.request
import zipfile

import pytest

from clip import *

def get_test_files():  # pragma: no cover
    """ Download some media files to use for the test, if they don't exist
    already."""

    if not os.path.exists("test_files"):
        os.mkdir("test_files")

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    def snag(fname, url):
        if not os.path.exists("test_files/" + fname):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(url, "test_files/" + fname)

    snag("books.mp4", "https://www.pexels.com/video/5224014/download")
    snag("music.mp3", "https://www.dropbox.com/s/mvvwaw1msplnteq/City%20Lights%20-%20The%20Lemming%20Shepherds.mp3?dl=1") #pylint: disable=line-too-long
    snag("water.png", "https://cdn.pixabay.com/photo/2017/09/14/11/07/water-2748640_1280.png")
    snag("flowers.png", "https://cdn.pixabay.com/photo/2017/02/11/17/08/flowers-2058090_1280.png")
    snag("bunny.webm", "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-webm-file.webm") # pylint: disable=line-too-long
    snag("snowman.pdf", "https://ctan.math.utah.edu/ctan/tex-archive/graphics/pgf/contrib/scsnowman/scsnowman-sample.pdf") # pylint: disable=line-too-long
    snag("brian.jpg", "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Brian_Wilson_%287314673472%29_%28tall%29.jpg/800px-Brian_Wilson_%287314673472%29_%28tall%29.jpg") # pylint: disable=line-too-long

    if not os.path.exists("test_files/bunny_frames"):
        os.mkdir("test_files/bunny_frames")
        ffmpeg('-i test_files/bunny.webm', 'test_files/bunny_frames/%04d.png')

    if not os.path.exists("test_files/bunny.zip"):
        with temporarily_changed_directory("test_files"):
            os.system("zip bunny.zip bunny_frames/*.png")

    exists = os.path.exists("test_files/ethnocentric_rg.ttf")
    exists = exists and os.path.exists("test_files/ethnocentric_rg_it.ttf")
    if not exists:
        with urllib.request.urlopen("https://dl.dafont.com/dl/?f=ethnocentric") as u:
            zip_data = u.read()
        file_like_object = io.BytesIO(zip_data)
        with zipfile.ZipFile(file_like_object) as z:
            with open("test_files/ethnocentric_rg.otf", 'wb') as f, \
              z.open("ethnocentric rg.otf") as ttf:
                f.write(ttf.read())
            with open("test_files/ethnocentric_rg_it.otf", 'wb') as f, \
              z.open("ethnocentric rg it.otf") as ttf:
                f.write(ttf.read())

def test_is_int():
    assert not is_int(0.3)
    assert is_int(3)
    assert not is_int("abc")
    assert not is_int(print)

    require_int(8, 'name')

def test_is_float():
    assert is_float(0.3)
    assert is_float(3)
    assert not is_float("abc")
    assert not is_float(print)

    require_float(8, 'name')
    require_float(7.7, 'name')

def test_is_color():
    assert is_color([0,0,0])
    assert not is_color([0,0,256])

    require_color([0,0,0], "color")
    with pytest.raises(TypeError):
        require_color(1024, "color")

def test_is_numbers():
    assert not is_positive(0)
    assert is_non_negative(0)
    assert is_non_negative(5)
    assert not is_non_negative(-2)
    assert is_even(2)
    assert not is_even(3)

    require_even(2, "the number")
    require_positive(4, 'the number')
    require_non_negative(0, 'the number')

def test_is_iterable():
    assert is_iterable([])
    assert not is_iterable(123)

def test_is_string():
    assert is_string("Tim")
    assert not is_string(123)
    require_string("Tim", "the string")

def test_is_equal():
    require_equal(1, 1, "name")

    with pytest.raises(ValueError):
        require_equal(1, "1", "name")

    with pytest.raises(ValueError):
        require_less_equal(2, 1, "two", "one")

def test_is_int_point():
    assert is_int_point((2, 3))
    assert not is_int_point((2.2, 3))
    assert not is_int_point("point")
    require_int_point((-6,7), "name")

def test_require_clip():
    with pytest.raises(TypeError):
        require_clip('Not a clip', 'name')

def test_require_less():
    with pytest.raises(ValueError):
        require_less(4, 3, 'first number', 'second number')

def test_require_callable():
    require_callable(print, 'print function')
    with pytest.raises(TypeError):
        require_callable('not a function', 'some string')

def test_ffmpeg():
    with pytest.raises(FFMPEGException), temporary_current_directory():
        ffmpeg('-i /dev/zero', '/dev/null')

    with pytest.raises(FFMPEGException), temporary_current_directory():
        ffmpeg('-i /dev/zero', '/dev/null', task="Testing", num_frames=100)

def test_metrics():
    m1 = Metrics(Clip.default_metrics)
    m2 = Metrics(Clip.default_metrics, length=0.5)

    with pytest.raises(ValueError):
        m1.verify_compatible_with(m2, check_length=True)

    with pytest.raises(TypeError):
        Metrics(Clip.default_metrics, width=0.5)

    with pytest.raises(ValueError):
        Metrics(Clip.default_metrics, width=-1)

    with pytest.raises(ValueError):
        Metrics(Clip.default_metrics, length=-1)

    with pytest.raises(ValueError):
        Metrics(Clip.default_metrics, length=0)

    with pytest.raises(TypeError):
        Metrics(Clip.default_metrics, length="really long")

def test_solid():
    x = solid([0,0,0], 640, 480, 300)
    x.verify(30)

    samples = x.get_samples()
    assert samples.shape == (x.num_samples(), x.num_channels())

def test_clip_metrics():
    secs = 30
    x = solid([0,0,0], 640, 480, secs)
    assert x.length() == secs
    assert x.sample_rate() == Clip.default_metrics.sample_rate
    assert x.num_samples() == secs*Clip.default_metrics.num_samples()
    assert x.num_channels() == Clip.default_metrics.num_channels

def test_verify1():
    # Valid
    x = solid([0,0,0], 640, 480, 10)
    x.verify(30)
    x.verify(0.1, verbose=True)

def test_verify2():
    # Return something that's not a frame.
    x = solid([0,0,0], 640, 480, 10)
    with pytest.raises(AssertionError):
        x.get_frame = lambda x: None
        x.verify(30)

def test_verify3():
    # Return the wrong size frame.
    x = solid([0,0,0], 640, 480, 10)
    with pytest.raises(ValueError):
        x.get_frame = lambda x: np.zeros([10, 10, 4], dtype=np.uint8)
        x.verify(30)

def test_readable_length1():
    x = solid([0,0,0], 640, 480, 30)
    assert ":30" in x.readable_length()

def test_readable_length2():
    x = solid([0,0,0], 640, 480, 60*60+1)
    assert x.readable_length()[:2] == '1:'

def test_sine_wave():
    x = sine_wave(880, 0.1, 5, 48000, 2)
    x.verify(20)

def test_mutator():
    a = black(640, 480, 5)
    b = MutatorClip(a)
    b.verify(30)

def test_scale_alpha():
    a = black(640, 480, 5)
    b = scale_alpha(a, 0.5)
    b.verify(30)

def test_black():
    black(640, 480, 300).verify(30)

def test_white():
    white(640, 480, 300).verify(30)

def test_read_image1():
    img = read_image('test_files/water.png')
    assert img.shape == (682, 1280, 4)
    assert img.dtype == np.uint8

def test_read_image2():
    img = read_image('test_files/brian.jpg')
    assert img.shape == (1067, 800, 4)
    assert img.dtype == np.uint8

def test_read_image3():
    with pytest.raises(FileNotFoundError):
        read_image("xyz.png")

def test_flatten_args():
    x = [[1, [2, 3]], 4]
    y = flatten_args(x)
    assert len(y) == 3

def test_sha256sum_file():
    h = sha256sum_file('test_files/brian.jpg')
    assert h[:7] == 'e16d354'

# Grab all of the test source files first.  (...instead of checking within each
# test.)
get_test_files()

