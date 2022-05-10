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
            with open("test_files/ethnocentric_rg.ttf", 'wb') as f, \
              z.open("ethnocentric rg.ttf") as ttf:
                f.write(ttf.read())
            with open("test_files/ethnocentric_rg_it.ttf", 'wb') as f, \
              z.open("ethnocentric rg_it.ttf") as ttf:
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

