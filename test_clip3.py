#!/usr/bin/env python3
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import

import glob
import io
import shutil
import sys
import urllib.request
import zipfile

import cv2
import pytest

from clip3 import *

def get_sample_files():  # pragma: no cover
    if not os.path.exists("samples"):
        os.mkdir("samples")

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    def snag(fname, url):
        if not os.path.exists("samples/" + fname):
            urllib.request.urlretrieve(url, "samples/" + fname)

    snag("books.mp4", "https://www.pexels.com/video/5224014/download")
    snag("music.mp3", "https://www.dropbox.com/s/mvvwaw1msplnteq/City%20Lights%20-%20The%20Lemming%20Shepherds.mp3?dl=1") #pylint: disable=line-too-long
    snag("water.png", "https://cdn.pixabay.com/photo/2017/09/14/11/07/water-2748640_1280.png")
    snag("flowers.png", "https://cdn.pixabay.com/photo/2017/02/11/17/08/flowers-2058090_1280.png")

    exists = os.path.exists("samples/ethnocentric_rg.ttf")
    exists = exists and os.path.exists("samples/ethnocentric_rg_it.ttf")
    if not exists:
        zip_data = urllib.request.urlopen("https://dl.dafont.com/dl/?f=ethnocentric").read()
        file_like_object = io.BytesIO(zip_data)
        with zipfile.ZipFile(file_like_object) as z:
            with open("samples/ethnocentric_rg.ttf", 'wb') as f:
                f.write(z.open("ethnocentric rg.ttf").read())
            with open("samples/ethnocentric_rg_it.ttf", 'wb') as f:
                f.write(z.open("ethnocentric rg it.ttf").read())

def test_validate():
    assert is_float(0.3)
    assert is_float(3)
    assert not is_float("abc")
    assert not is_float(print)

    assert is_color([0,0,0])
    assert not is_color([0,0,256])

    assert not is_positive(0)
    assert is_non_negative(0)
    assert is_non_negative(5)
    assert not is_non_negative(-2)

    assert is_iterable(list())
    assert not is_iterable(123)

    assert is_string("Tim")
    assert not is_string(123)
    require_string("Tim", "the string")

    require_equal(1, 1, "name")

    with pytest.raises(ValueError):
        require_equal(1, "1", "name")

    with pytest.raises(ValueError):
        require_less_equal(2, 1, "two", "one")

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
    x = solid([0,0,0], 640, 480, 30, 300)
    x.verify()

    samples = x.get_samples()
    assert samples.shape == (x.num_samples(), x.num_channels())

def test_clip_metrics():
    secs = 30
    x = solid([0,0,0], 640, 480, 30, secs)
    assert x.length() == secs
    assert x.frame_rate() == 30
    assert x.sample_rate() == default_metrics.sample_rate
    assert x.num_samples() == secs*default_metrics.num_samples()
    assert x.num_frames() == secs*30
    assert x.num_channels() == default_metrics.num_channels
    assert f":{secs:02d}" in x.readable_length()

    x = solid([0,0,0], 640, 480, 30, 60*60+1)
    assert x.readable_length()[:2] == '1:'

def test_temporary_current_directory():
    with temporary_current_directory():
        assert glob.glob('*') == []

def test_cache():
    c = ClipCache()
    c.clear()

    x = solid([0,0,0], 640, 480, 30, 300)
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
    x = solid([0,0,0], 640, 480, 30, secs)

    assert x.frame_to_sample(0) == 0
    assert x.frame_to_sample(x.frame_rate()) == x.sample_rate()

def test_ffmpeg():
    with pytest.raises(FFMPEGException), temporary_current_directory():
        ffmpeg('-i /dev/zero', '/dev/null')

    with pytest.raises(FFMPEGException), temporary_current_directory():
        ffmpeg('-i /dev/zero', '/dev/null', task="Testing", num_frames=100)

def test_save():
    shutil.rmtree(cache.directory)
    cache.cache = None

    x = solid([0,0,0], 640, 480, 30, 10)
    with temporary_current_directory():
        x.save('test.mp4')
        assert os.path.exists('test.mp4')

    with temporary_current_directory():
        x.save('foo.flac')
        x.save('foo.wav')
        assert os.path.exists('foo.flac')
        assert os.path.exists('foo.wav')

def test_composite():
    x = solid([0,0,0], 640, 480, 30, 5)
    y = solid([0,0,0], 640, 481, 30, 5)

    with pytest.raises(ValueError):
        # Heights do not match.
        z = composite(
          Element(x, 0, [0, 0]),
          Element(y, 6, [0, 0])
        )

    x = solid([0,0,0], 640, 480, 30, 5)
    y = solid([0,0,0], 640, 480, 30, 5)

    with pytest.raises(ValueError):
        # Can't start before 0.
        z = composite(
          Element(x, -1, [0, 0]),
          Element(y, 6, [0, 0])
        )

    x = solid([0,0,0], 640, 480, 30, 5)
    y = solid([0,0,0], 640, 480, 31, 5)

    with pytest.raises(ValueError):
        # Frame rates don't match.
        z = composite(
          Element(x, -1, [0, 0]),
          Element(y, 6, [0, 0])
        )

    x = sine_wave(880, 0.1, 5, 48000, 2)
    y = sine_wave(880, 0.1, 5, 48001, 2)

    with pytest.raises(ValueError):
        # Sample rates don't match.
        z = composite(
          Element(x, 0, [0, 0]),
          Element(y, 5, [0, 0])
        )

    x = solid([0,0,0], 640, 480, 30, 5)
    y = solid([0,0,0], 640, 480, 30, 5)
    z = composite(
      Element(x, 0, [0, 0]),
      Element(y, 6, [0, 0])
    )
    assert z.length() == 11
    z.verify()

def test_sine_wave():
    x = sine_wave(880, 0.1, 5, 48000, 2)
    x.verify()

def test_join():
    x = sine_wave(440, 0.25, 5, 48000, 2)
    y = solid([0,255,0], 640, 480, 30, 5)
    z = join(y, x)
    z.verify()

    with pytest.raises(AssertionError):
        join(x, y)


def test_chain_and_fade_chain():
    a = black(640, 480, 30, 3)
    b = white(640, 480, 30, 3)
    c = solid([255,0,0], 640, 480, 30, 3)

    d = chain(a, [b, c])
    assert d.length() == a.length() + b.length() + c.length()
    d.verify()

    e = fade_chain(2, a, [b, c])
    assert e.length() == a.length() + b.length() + c.length() - 4
    e.verify()

    with pytest.raises(ValueError):
        chain()

    with pytest.raises(ValueError):
        fade_chain(3)


def test_black_and_white():
    black(640, 480, 30, 300).verify()
    white(640, 480, 30, 300).verify()

def test_mutator():
    a = black(640, 480, 30, 5)
    b = MutatorClip(a)
    b.verify()

def test_scale_alpha():
    a = black(640, 480, 30, 5)
    b = scale_alpha(a, 0.5)
    b.verify()


def test_preview():
    cache.clear()
    x = solid([0,0,0], 640, 480, 30, 5)
    x.preview()

def test_metrics_from_ffprobe_output():
    video_deets = "stream|index=0|codec_name=h264|codec_long_name=H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10|profile=High|codec_type=video|codec_time_base=1/60|codec_tag_string=avc1|codec_tag=0x31637661|width=1024|height=576|coded_width=1024|coded_height=576|has_b_frames=2|sample_aspect_ratio=N/A|display_aspect_ratio=N/A|pix_fmt=yuv420p|level=32|color_range=unknown|color_space=unknown|color_transfer=unknown|color_primaries=unknown|chroma_location=left|field_order=unknown|timecode=N/A|refs=1|is_avc=true|nal_length_size=4|id=N/A|r_frame_rate=30/1|avg_frame_rate=30/1|time_base=1/15360|start_pts=0|start_time=0.000000|duration_ts=1416192|duration=92.200000|bit_rate=1134131|max_bit_rate=N/A|bits_per_raw_sample=8|nb_frames=2766|nb_read_frames=N/A|nb_read_packets=N/A|disposition:default=1|disposition:dub=0|disposition:original=0|disposition:comment=0|disposition:lyrics=0|disposition:karaoke=0|disposition:forced=0|disposition:hearing_impaired=0|disposition:visual_impaired=0|disposition:clean_effects=0|disposition:attached_pic=0|disposition:timed_thumbnails=0|tag:language=und|tag:handler_name=VideoHandler" # pylint: disable=line-too-long
    audio_deets = "stream|index=1|codec_name=aac|codec_long_name=AAC (Advanced Audio Coding)|profile=LC|codec_type=audio|codec_time_base=1/44100|codec_tag_string=mp4a|codec_tag=0x6134706d|sample_fmt=fltp|sample_rate=44100|channels=2|channel_layout=stereo|bits_per_sample=0|id=N/A|r_frame_rate=0/0|avg_frame_rate=0/0|time_base=1/44100|start_pts=0|start_time=0.000000|duration_ts=4066020|duration=92.200000|bit_rate=128751|max_bit_rate=128751|bits_per_raw_sample=N/A|nb_frames=3972|nb_read_frames=N/A|nb_read_packets=N/A|disposition:default=1|disposition:dub=0|disposition:original=0|disposition:comment=0|disposition:lyrics=0|disposition:karaoke=0|disposition:forced=0|disposition:hearing_impaired=0|disposition:visual_impaired=0|disposition:clean_effects=0|disposition:attached_pic=0|disposition:timed_thumbnails=0|tag:language=und|tag:handler_name=SoundHandler" # pylint: disable=line-too-long
    short_audio_deets = "stream|index=1|codec_name=aac|codec_long_name=AAC (Advanced Audio Coding)|profile=LC|codec_type=audio|codec_time_base=1/44100|codec_tag_string=mp4a|codec_tag=0x6134706d|sample_fmt=fltp|sample_rate=44100|channels=2|channel_layout=stereo|bits_per_sample=0|id=N/A|r_frame_rate=0/0|avg_frame_rate=0/0|time_base=1/44100|start_pts=0|start_time=0.000000|duration_ts=4066020|duration=91.000000|bit_rate=128751|max_bit_rate=128751|bits_per_raw_sample=N/A|nb_frames=3972|nb_read_frames=N/A|nb_read_packets=N/A|disposition:default=1|disposition:dub=0|disposition:original=0|disposition:comment=0|disposition:lyrics=0|disposition:karaoke=0|disposition:forced=0|disposition:hearing_impaired=0|disposition:visual_impaired=0|disposition:clean_effects=0|disposition:attached_pic=0|disposition:timed_thumbnails=0|tag:language=und|tag:handler_name=SoundHandler" # pylint: disable=line-too-long
    bogus_deets = "stream|index=1|codec_name=aac|profile=LC|codec_type=trash"

    correct_metrics = Metrics(
      width=1024,
      height=576,
      frame_rate=30.0,
      sample_rate=44100,
      num_channels=2,
      length=92.2
    )

    with pytest.raises(ValueError):
        metrics_from_ffprobe_output(f'{video_deets}\n{video_deets}', 'test.mp4')
    with pytest.raises(ValueError):
        metrics_from_ffprobe_output(f'{audio_deets}\n{audio_deets}', 'test.mp4')
    with pytest.raises(ValueError):
        metrics_from_ffprobe_output(f'{audio_deets}\n{video_deets}\n{video_deets}', 'test.mp4')
    with pytest.raises(ValueError):
        metrics_from_ffprobe_output(f'{audio_deets}\n{video_deets}\n{bogus_deets}', 'test.mp4')
    with pytest.raises(ValueError):
        metrics_from_ffprobe_output(f'{short_audio_deets}\n{video_deets}', 'test.mp4')
    with pytest.raises(ValueError):
        metrics_from_ffprobe_output('', 'test.mp4')

    m, _, _ = metrics_from_ffprobe_output(f'{audio_deets}\n{video_deets}', 'test.mp4')
    assert m == correct_metrics

    m, _, _ = metrics_from_ffprobe_output(f'{video_deets}\n{audio_deets}', 'test.mp4')
    assert m == correct_metrics

    m, _, _ = metrics_from_ffprobe_output(f'{video_deets}', 'test.mp4')
    assert m == Metrics(
      src=correct_metrics,
      sample_rate=default_metrics.sample_rate,
      num_channels=default_metrics.num_channels
    )

    m, _, _ = metrics_from_ffprobe_output(f'{audio_deets}', 'test.mp4')
    assert m == Metrics(
      src=correct_metrics,
      width=default_metrics.width,
      height=default_metrics.height,
      frame_rate=default_metrics.frame_rate,
    )

def test_from_file():
    get_sample_files()
    with pytest.raises(FileNotFoundError):
        from_file("samples/books12312312.mp4")

    cache.clear()
    b = from_file("samples/books.mp4", forced_length=2)
    b.verify()
    assert b.length() == 2


    a = from_file("samples/books.mp4", decode_chunk_length=None)
    a.verify()

    c = from_file("samples/music.mp3")
    c.verify()

def test_audio_samples_from_file():
    get_sample_files()

    with pytest.raises(FFMPEGException):
        # No audio track.
        audio_samples_from_file(
          "samples/books.mp4",
          expected_num_samples=0,
          expected_num_channels=1,
          expected_sample_rate=0
        )

    with pytest.raises(ValueError):
        # Wrong sample rate.
        audio_samples_from_file(
          "samples/music.mp3",
          expected_num_samples=3335168,
          expected_num_channels=2,
          expected_sample_rate=48000
        )

    with pytest.raises(ValueError):
        # Wrong number of channels
        audio_samples_from_file(
          "samples/music.mp3",
          expected_num_samples=3335168,
          expected_num_channels=1,
          expected_sample_rate=44100
        )

    with pytest.raises(ValueError):
        # Wrong length.
        audio_samples_from_file(
          "samples/music.mp3",
          expected_num_samples=4335170,
          expected_num_channels=2,
          expected_sample_rate=44100
        )

    # # Slightly too long.
    audio_samples_from_file(
      "samples/music.mp3",
      expected_num_samples=3337343,
      expected_num_channels=2,
      expected_sample_rate=44100
    )

    # Slightly too short.
    audio_samples_from_file(
      "samples/music.mp3",
      expected_num_samples=3337345,
      expected_num_channels=2,
      expected_sample_rate=44100
    )

    # All good.
    audio_samples_from_file(
      "samples/music.mp3",
      expected_num_samples=3337344,
      expected_num_channels=2,
      expected_sample_rate=44100
    )

def test_slice_clip():
    a = join(
      solid([0,0,0], 640, 480, 30, 10),
      sine_wave(880, 0.1, 10, 48000, 2)
    )

    with pytest.raises(ValueError):
        slice_clip(a, -1, 1)

    with pytest.raises(ValueError):
        slice_clip(a, 3, 1)


    d = slice_clip(a, 3, 4)
    d.verify()

    e = slice_clip(a, 3)
    e.verify()

    f = slice_clip(a, end=3)
    f.verify()

def test_mono_to_stereo():
    a = sine_wave(880, 0.1, 10, 48000, 1)
    b = mono_to_stereo(a)
    b.verify()
    assert b.num_channels() == 2

    a = sine_wave(880, 0.1, 10, 48000, 2)
    with pytest.raises(ValueError):
        # Not a mono source.
        b = mono_to_stereo(a)

def test_stereo_to_mono():
    a = sine_wave(880, 0.1, 10, 48000, 2)
    b = stereo_to_mono(a)
    b.verify()
    assert b.num_channels() == 1

    a = sine_wave(880, 0.1, 10, 48000, 1)
    with pytest.raises(ValueError):
        # Not a stereo source.
        b = stereo_to_mono(a)

def test_reverse():
    a = join(
      solid([0,0,0], 640, 480, 30, 10),
      sine_wave(880, 0.1, 10, 48000, 2)
    )
    b = join(
      solid([255,0,0], 640, 480, 30, 10),
      sine_wave(440, 0.1, 10, 48000, 2)
    )
    c = chain(a,b)
    d = reverse(c)
    d.verify()

    f1 = c.get_frame(5)
    f2 = d.get_frame(d.num_frames()-6)
    assert np.array_equal(f1, f2)

    s1 = c.get_samples()
    s2 = d.get_samples()

    assert np.array_equal(s1[5], s2[s2.shape[0]-6])

def test_volume():
    a = sine_wave(880, 0.1, 10, 48000, 2)
    b = scale_volume(a, 0.1)
    b.verify()

    with pytest.raises(ValueError):
        scale_volume(a, -10)

    with pytest.raises(TypeError):
        scale_volume(a, "tim")

def test_crop():
    a = join(
      solid([0,0,0], 640, 480, 30, 10),
      sine_wave(880, 0.1, 10, 48000, 2)
    )

    b = crop(a, [10, 10], [100, 100])
    b.verify()
    assert b.width() == 90
    assert b.height() == 90

    with pytest.raises(ValueError):
        crop(a, [-1, 10], [100, 100])

    with pytest.raises(ValueError):
        crop(a, [100, 100], [10, 10])

    with pytest.raises(ValueError):
        crop(a, [10, 10], [100, 10000])

def test_get_font():
    get_sample_files()

    get_font("samples/ethnocentric_rg.ttf", 10)
    get_font("samples/ethnocentric_rg.ttf", 10)
    get_font("samples/ethnocentric_rg_it.ttf", 20)

    with pytest.raises(ValueError):
        get_font("clip3.py", 20)

    with pytest.raises(ValueError):
        get_font("samples/asdasdasdsad.ttf", 20)

def test_draw_text():
    get_sample_files()
    font = "samples/ethnocentric_rg_it.ttf"
    x = draw_text("Hello!", font, font_size=200, frame_rate=30, length=5)
    x.verify()


def test_alpha_blend():
    get_sample_files()

    f0 = cv2.imread("samples/flowers.png", cv2.IMREAD_UNCHANGED)
    f1 = np.zeros(shape=f0.shape, dtype=np.uint8)
    f2 = alpha_blend(f0, f1)
    cv2.imwrite('samples/blended.png', f2)

    f0 = cv2.imread("samples/water.png", cv2.IMREAD_UNCHANGED)
    f0 = f0[0:439,:,:]
    f1 = cv2.imread("samples/flowers.png", cv2.IMREAD_UNCHANGED)
    f2 = alpha_blend(f0, f1)

def test_to_monochrome():
    a = black(640, 480, 30, 3)
    b = to_monochrome(a)
    b.verify()

def test_filter_frames():

    a = black(640, 480, 30, 3)

    b = filter_frames(a, lambda x: x)
    b.verify()

    c = filter_frames(a, lambda x: x, name='identity')
    c.verify()

    d = filter_frames(a, lambda x: x, size='same')
    d.verify()

    e = filter_frames(a, lambda x: x, size=(a.width(), a.height()))
    e.verify()

    # Nonsense size
    with pytest.raises(ValueError):
        filter_frames(a, lambda x: x, size='sooom')

    # Wrong size
    f = filter_frames(a, lambda x: x, size=(10, 10))
    with pytest.raises(ValueError):
        f.verify()

    g = filter_frames(a, lambda x: x)
    h = filter_frames(a, lambda x: x)
    i = filter_frames(a, lambda x: x+1)
    assert g.sig == h.sig
    assert h.sig != i.sig


def test_scale_to_size():
    a = black(640, 480, 30, 3)
    b = scale_to_size(a, 100, 200) 
    b.verify()
    assert b.width() == 100
    assert b.height() == 200



# If we're run as a script, just execute some or all of the tests.
if __name__ == '__main__':  #pragma: no cover
    try:
        pattern = sys.argv[1]
    except IndexError:
        pattern = ""
    for name, thing in list(globals().items()):
        if 'test_' in name and pattern in name:
            thing()

