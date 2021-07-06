#!/usr/bin/env python3
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import
# pylint: disable=too-many-lines

import glob
import io
import sys
import urllib.request
import zipfile

import cv2
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
        zip_data = urllib.request.urlopen("https://dl.dafont.com/dl/?f=ethnocentric").read()
        file_like_object = io.BytesIO(zip_data)
        with zipfile.ZipFile(file_like_object) as z:
            with open("test_files/ethnocentric_rg.ttf", 'wb') as f:
                f.write(z.open("ethnocentric rg.ttf").read())
            with open("test_files/ethnocentric_rg_it.ttf", 'wb') as f:
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
    assert is_even(2)
    assert not is_even(3)

    require_even(2, "the number")

    assert is_iterable(list())
    assert not is_iterable(123)

    assert is_string("Tim")
    assert not is_string(123)
    require_string("Tim", "the string")

    require_equal(1, 1, "name")

    require_color([0,0,0], "color")
    with pytest.raises(TypeError):
        require_color(1024, "color")

    with pytest.raises(ValueError):
        require_equal(1, "1", "name")

    with pytest.raises(ValueError):
        require_less_equal(2, 1, "two", "one")

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
    x = solid([0,0,0], 640, 480, 30, 300)
    x.verify()

    test_files = x.get_samples()
    assert test_files.shape == (x.num_samples(), x.num_channels())

def test_clip_metrics():
    secs = 30
    x = solid([0,0,0], 640, 480, 30, secs)
    assert x.length() == secs
    assert x.frame_rate() == 30
    assert x.sample_rate() == Clip.default_metrics.sample_rate
    assert x.num_samples() == secs*Clip.default_metrics.num_samples()
    assert x.num_frames() == secs*30
    assert x.num_channels() == Clip.default_metrics.num_channels
    assert f":{secs:02d}" in x.readable_length()

    x = solid([0,0,0], 640, 480, 30, 60*60+1)
    assert x.readable_length()[:2] == '1:'

def test_clip_metrics2():
    # A fractional frame at the end.
    a = black(640, 480, 1, 1.25)
    assert a.num_frames()==2

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
    cache.clear()

    x = solid([0,0,0], 640, 480, 30, 10)
    with temporary_current_directory():
        x.save('test.mp4')
        assert os.path.exists('test.mp4')

    with temporary_current_directory():
        x.save('foo.flac')
        x.save('foo.wav')
        assert os.path.exists('foo.flac')
        assert os.path.exists('foo.wav')

def test_composite1():
    # For automatically computing the height.
    x = solid([0,0,0], 640, 480, 30, 5)
    y = solid([0,0,0], 640, 481, 30, 5)
    z = composite(
      Element(x, 0, [0, 0]),
      Element(y, 6, [0, 0])
    )
    z.verify()
    print(z.metrics)
    assert z.height() == 481

def test_composite2():
    # Can't start before 0.
    x = solid([0,0,0], 640, 480, 30, 5)
    y = solid([0,0,0], 640, 480, 30, 5)

    with pytest.raises(ValueError):
        composite(
          Element(x, -1, [0, 0]),
          Element(y, 6, [0, 0])
        )

def test_composite3():
    # Frame rates don't match.
    x = solid([0,0,0], 640, 480, 30, 5)
    y = solid([0,0,0], 640, 480, 31, 5)
    with pytest.raises(ValueError):
        composite(
          Element(x, -1, [0, 0]),
          Element(y, 6, [0, 0])
        )

def test_composite4():
    # Sample rates don't match.
    x = sine_wave(880, 0.1, 5, 48000, 2)
    y = sine_wave(880, 0.1, 5, 48001, 2)

    with pytest.raises(ValueError):
        composite(
          Element(x, 0, [0, 0]),
          Element(y, 5, [0, 0])
        )

def test_composite5():
    # Automatically computed length.
    x = solid([0,0,0], 640, 480, 30, 5)
    y = solid([0,0,0], 640, 480, 30, 5)
    z = composite(
      Element(x, 0, [0, 0]),
      Element(y, 6, [0, 0])
    )
    assert z.length() == 11
    z.verify()

def test_composite6():
    # Clipping above, below, left, and right.
    x = static_image("test_files/flowers.png", 30, 10)
    x = scale_by_factor(x, 0.4)

    z = composite(
      Element(x, 0, [-100, -100], VideoMode.REPLACE),
      Element(x, 0, [540, 380], VideoMode.REPLACE),
      width=640,
      height=480,
      length=5
    )
    z.verify()

def test_composite7():
    # Totally off-screen.
    x = static_image("test_files/flowers.png", 30, 10)
    x = scale_by_factor(x, 0.4)

    z = composite(
      Element(x, 0, [-1000, -100], VideoMode.REPLACE),
      Element(x, 0, [1540, 380], VideoMode.REPLACE),
      width=640,
      height=480,
      length=5
    )
    z.verify()


def test_composite8():
    # Alpha blending.
    x = static_image("test_files/flowers.png", 30, 5000)
    x = scale_by_factor(x, 0.4)

    z = composite(
      Element(x, 0, [50, 50], video_mode=VideoMode.BLEND),
      Element(x, 0, [250, 150], video_mode=VideoMode.BLEND),
      width=640,
      height=480,
      length=1
    )
    z.verify()

def test_composite9():
    # Bad inputs.
    x = static_image("test_files/flowers.png", 30, 5000)
    with pytest.raises(ValueError):
        # Bad position, iterable but wrong length.
        composite(Element(x, 0, [0,0,0], video_mode=VideoMode.BLEND))

    with pytest.raises(TypeError):
        # Bad position, iterable but not ints.
        composite(Element(x, 0, "ab", video_mode=VideoMode.BLEND))

    with pytest.raises(TypeError):
        # Bad position, not even iterable.
        composite(Element(x, 0, 0, video_mode=VideoMode.BLEND))

    with pytest.raises(TypeError):
        # Bad video mode.
        composite(Element(x, 0, [0, 0], video_mode=AudioMode.REPLACE))

    with pytest.raises(TypeError):
        # Bad audio mode.
        composite(Element(x, 0, [0, 0], audio_mode=VideoMode.REPLACE))

def test_composite10():
    # Callable position.
    x = static_image("test_files/flowers.png", 30, 5)
    x = scale_by_factor(x, 0.4)

    def pos1(index):
        return [index-100,2*index-100]
    def pos2(index):
        return [480-index,2*index-100]

    z = composite(
      Element(x, 0, pos1, video_mode=VideoMode.BLEND),
      Element(x, 0, pos2, video_mode=VideoMode.BLEND),
      length=5
    )
    z.verify()

def test_composite11():
    # Ignored video should not impact the frame signatures.
    a = static_image("test_files/flowers.png", 30, 5)

    b = composite(
      Element(a, 0, [0,0], video_mode=VideoMode.BLEND),
    )

    c = composite(
      Element(a, 0, [0,0], video_mode=VideoMode.BLEND),
      Element(a, 0, [10,10], video_mode=VideoMode.IGNORE),
    )

    assert b.frame_signature(0) == c.frame_signature(0)

def test_sine_wave():
    x = sine_wave(880, 0.1, 5, 48000, 2)
    x.verify()

def test_join():
    x = sine_wave(440, 0.25, 3, 48000, 2)
    y = solid([0,255,0], 640, 480, 30, 5)
    z = join(y, x)
    z.verify()
    assert y.length() == 5

    # Complain if we can detect that audio and video are swapped.
    with pytest.raises(AssertionError):
        join(x, y)


def test_chain():
    a = black(640, 480, 30, 3)
    b = white(640, 480, 30, 3)
    c = solid([255,0,0], 640, 480, 30, 3)

    d = chain(a, [b, c])
    assert d.length() == a.length() + b.length() + c.length()
    d.verify()

    e = chain(a, [b, c], fade_time=2)
    assert e.length() == a.length() + b.length() + c.length() - 4
    e.verify()

    with pytest.raises(ValueError):
        chain()

    with pytest.raises(ValueError):
        chain(fade_time=3)


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

def test_metrics_from_ffprobe_output1():
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

    with pytest.raises(ValueError):
        bad_video_deets = re.sub("duration", "dooration", video_deets)
        metrics_from_ffprobe_output(f'{bad_video_deets}\n{audio_deets}', 'test.mp4')

    m, _, _ = metrics_from_ffprobe_output(f'{audio_deets}\n{video_deets}', 'test.mp4')
    assert m == correct_metrics

    m, _, _ = metrics_from_ffprobe_output(f'{video_deets}\n{audio_deets}', 'test.mp4')
    assert m == correct_metrics

    m, _, _ = metrics_from_ffprobe_output(f'{video_deets}', 'test.mp4')
    assert m == Metrics(
      src=correct_metrics,
      sample_rate=Clip.default_metrics.sample_rate,
      num_channels=Clip.default_metrics.num_channels
    )

    m, _, _ = metrics_from_ffprobe_output(f'{audio_deets}', 'test.mp4')
    assert m == Metrics(
      src=correct_metrics,
      width=Clip.default_metrics.width,
      height=Clip.default_metrics.height,
      frame_rate=Clip.default_metrics.frame_rate,
    )

def test_metrics_from_ffprobe_output2():
    rotated_video_deets = "stream|index=0|codec_name=h264|codec_long_name=H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10|profile=Baseline|codec_type=video|codec_time_base=18821810/1129461271|codec_tag_string=avc1|codec_tag=0x31637661|width=1600|height=1200|coded_width=1600|coded_height=1200|has_b_frames=0|sample_aspect_ratio=1:1|display_aspect_ratio=4:3|pix_fmt=yuvj420p|level=10|color_range=pc|color_space=smpte170m|color_transfer=smpte170m|color_primaries=bt470bg|chroma_location=left|field_order=unknown|timecode=N/A|refs=1|is_avc=true|nal_length_size=4|id=N/A|r_frame_rate=30/1|avg_frame_rate=1129461271/37643620|time_base=1/90000|start_pts=0|start_time=0.000000|duration_ts=128477601|duration=1427.528900|bit_rate=18000964|max_bit_rate=N/A|bits_per_raw_sample=8|nb_frames=42832|nb_read_frames=N/A|nb_read_packets=N/A|disposition:default=1|disposition:dub=0|disposition:original=0|disposition:comment=0|disposition:lyrics=0|disposition:karaoke=0|disposition:forced=0|disposition:hearing_impaired=0|disposition:visual_impaired=0|disposition:clean_effects=0|disposition:attached_pic=0|disposition:timed_thumbnails=0|tag:rotate=90|tag:creation_time=2020-08-18T15:50:05.000000Z|tag:language=eng|tag:handler_name=VideoHandle" #pylint: disable=line-too-long
    m, _, _ = metrics_from_ffprobe_output(f'{rotated_video_deets}', 'test.mp4')
    print(m)

def test_from_file1():
    with pytest.raises(FileNotFoundError):
        from_file("test_files/books12312312.mp4")

def test_from_file2():
    cache.clear()
    a = from_file("test_files/bunny.webm", decode_chunk_length=1.0)
    a = slice_clip(a, 0, 1.1)
    a.verify()

    b = from_file("test_files/bunny.webm", decode_chunk_length=None)
    b.verify()

    # Again to use the cached dimensions.
    c = from_file("test_files/bunny.webm", decode_chunk_length=None)
    c.verify()

def test_from_file3():
    cache.clear()

    # For the case with no video.
    d = from_file("test_files/music.mp3")
    d.verify()

def test_from_file4():
    cache.clear()

    # For the case with no audio.
    e = from_file("test_files/books.mp4", decode_chunk_length=1.0)
    e = slice_clip(e, 0, 0.5)
    e.verify()

def test_from_file5():
    # Suppress audio and suppress video.
    a = from_file("test_files/bunny.webm", suppress_audio=True)
    assert a.has_audio is False

    b = from_file("test_files/bunny.webm", suppress_video=True)
    assert b.has_audio is True

def test_audio_samples_from_file():
    with pytest.raises(FFMPEGException):
        # No audio track.
        audio_samples_from_file(
          "test_files/books.mp4",
          expected_num_samples=0,
          expected_num_channels=1,
          expected_sample_rate=0
        )

    with pytest.raises(ValueError):
        # Wrong sample rate.
        audio_samples_from_file(
          "test_files/music.mp3",
          expected_num_samples=3335168,
          expected_num_channels=2,
          expected_sample_rate=48000
        )

    with pytest.raises(ValueError):
        # Wrong number of channels
        audio_samples_from_file(
          "test_files/music.mp3",
          expected_num_samples=3335168,
          expected_num_channels=1,
          expected_sample_rate=44100
        )

    with pytest.raises(ValueError):
        # Wrong length.
        audio_samples_from_file(
          "test_files/music.mp3",
          expected_num_samples=4335170,
          expected_num_channels=2,
          expected_sample_rate=44100
        )

    # # Slightly too long.
    audio_samples_from_file(
      "test_files/music.mp3",
      expected_num_samples=3337343,
      expected_num_channels=2,
      expected_sample_rate=44100
    )

    # Slightly too short.
    audio_samples_from_file(
      "test_files/music.mp3",
      expected_num_samples=3337345,
      expected_num_channels=2,
      expected_sample_rate=44100
    )

    # All good.
    audio_samples_from_file(
      "test_files/music.mp3",
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
      solid([0,0,0], 640, 480, 30, 1),
      sine_wave(880, 0.1, 1, 48000, 2)
    )
    b = join(
      solid([255,0,0], 640, 480, 30, 1),
      sine_wave(440, 0.1, 1, 48000, 2)
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

    # Typical usage.
    b = crop(a, [10, 10], [100, 100])
    b.verify()
    assert b.width() == 90
    assert b.height() == 90


    # Zero is okay.
    c = crop(a, [0, 0], [100, 100])
    c.verify()

    with pytest.raises(ValueError):
        crop(a, [-1, 10], [100, 100])

    with pytest.raises(ValueError):
        crop(a, [100, 100], [10, 10])

    with pytest.raises(ValueError):
        crop(a, [10, 10], [100, 10000])

def test_get_font():
    get_font("test_files/ethnocentric_rg.ttf", 10)
    get_font("test_files/ethnocentric_rg.ttf", 10)
    get_font("test_files/ethnocentric_rg_it.ttf", 20)

    with pytest.raises(ValueError):
        get_font("clip3.py", 20)

    with pytest.raises(ValueError):
        get_font("test_files/asdasdasdsad.ttf", 20)

def test_draw_text():
    font = "test_files/ethnocentric_rg_it.ttf"
    x = draw_text("Hello!", font, font_size=200, color=[255,0,255], frame_rate=30, length=5)
    x.verify()


def test_alpha_blend():
    f0 = cv2.imread("test_files/flowers.png", cv2.IMREAD_UNCHANGED)
    f1 = np.zeros(shape=f0.shape, dtype=np.uint8)
    f2 = alpha_blend(f0, f1)
    cv2.imwrite('test_files/blended.png', f2)

    f0 = cv2.imread("test_files/water.png", cv2.IMREAD_UNCHANGED)
    f0 = f0[0:439,:,:]
    f1 = cv2.imread("test_files/flowers.png", cv2.IMREAD_UNCHANGED)
    f2 = alpha_blend(f0, f1)

def test_to_monochrome():
    a = black(640, 480, 30, 3)
    b = to_monochrome(a)
    b.verify()

def test_filter_frames1():
    a = black(640, 480, 30, 3)

    b = filter_frames(a, lambda x: x)
    assert not b.depends_on_index
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

    # Signatures match?
    g = filter_frames(a, lambda x: x)
    h = filter_frames(a, lambda x: x)
    i = filter_frames(a, lambda x: x+1)
    assert g.sig == h.sig
    assert h.sig != i.sig

def test_filter_frames2():
    # Two-parameter filter version.
    a = black(640, 480, 30, 3)
    b = filter_frames(a, lambda x, i: x)
    b.verify()

def test_filter_frames3():
    # A bogus filter.
    a = black(640, 480, 30, 3)
    with pytest.raises(TypeError):
        filter_frames(a, lambda x, y, z: None)

def test_scale_to_size():
    a = black(640, 480, 30, 3)
    b = scale_to_size(a, 100, 200)
    b.verify()
    assert b.width() == 100
    assert b.height() == 200

def test_scale_by_factor():
    a = black(100, 200, 30, 3)
    b = scale_by_factor(a, 0.1)
    b.verify()
    assert b.width() == 10
    assert b.height() == 20

def test_scale_to_fit():
    a = black(100, 100, 30, 3)
    b = scale_to_fit(a, 50, 100)
    b.verify()
    assert abs(b.width()/b.height() - 1.0)  < 1e-10

    c = scale_to_fit(a, 100, 50)
    c.verify()
    assert abs(b.width()/b.height() - 1.0)  < 1e-10

def test_static_frame1():
    # Legit usage: An RGBA image.
    a = static_image("test_files/water.png", 30, 10)
    a.verify()

def test_static_frame2():
    # Legit usage: An RGB image.
    b = static_image("test_files/brian.jpg", 30, 10)
    b.verify()


def test_static_frame3():
    # Wrong type
    with pytest.raises(TypeError):
        static_frame("not a frame", "name", 30, 10)

def test_static_frame4():
    # Wrong shape
    with pytest.raises(ValueError):
        static_frame(np.zeros([100, 100]), "name", 30, 10)

def test_static_frame5():
    # Wrong number of channels
    with pytest.raises(ValueError):
        static_frame(np.zeros([100, 100, 3]), "name", 30, 10)

def test_static_frame6():
    # Frame signatures should depend (only) on the contents; same contents give
    # same frame signature.
    black_frame = np.zeros([100, 200, 4], np.uint8)
    black_frame[:] = [0, 0, 0, 255]

    black_frame2 = np.zeros([100, 200, 4], np.uint8)
    black_frame2[:] = [0, 0, 0, 255]

    mostly_black_frame = np.zeros([100, 200, 4], np.uint8)
    mostly_black_frame[:] = [0, 0, 0, 255]
    mostly_black_frame[50,50,1] = 6

    white_frame = np.zeros([100, 200, 4], np.uint8)
    white_frame[:] = [255, 255, 255, 255]

    a = static_frame(black_frame, "blackness", 30, 3)
    b = static_frame(black_frame2, "blackness", 30, 3)
    c = static_frame(white_frame, "whiteness", 30, 3)
    d = static_frame(mostly_black_frame, "mostly black", 30, 3)

    assert a.sig == b.sig
    assert a.sig != c.sig
    assert a.sig != d.sig



def test_resample1():
    # Basic case.
    length = 5
    a = from_file("test_files/bunny.webm", decode_chunk_length=length)
    a = slice_clip(a, 0, length)

    fr = 29
    sr = 48000
    l = 2*length
    b = resample(a, frame_rate=fr, sample_rate=sr, length=l)
    assert b.frame_rate() == fr
    assert b.sample_rate() == sr
    assert b.length() == l
    b.verify()

def test_resample2():
    # Cover all of the default-parameter branches.
    length = 5
    a = from_file("test_files/bunny.webm", decode_chunk_length=length)
    a = slice_clip(a, 0, length)

    b = resample(a)
    b.verify()

def test_resample3():
    # Ensure that frames are being resampled correctly.
    length = 5
    a = from_file("test_files/bunny.webm", decode_chunk_length=length)
    a = slice_clip(a, 0, length)

    fr = a.frame_rate()/2
    print("fr=", fr)
    b = resample(a, frame_rate=fr)
    b.verify(verbose=True)
    assert b.frame_rate() == a.frame_rate()/2
    assert b.new_index(0) == 0

def test_fades():
    cache.clear()
    a = white(640, 480, 30, 3)

    for cls in [fade_in, fade_out]:
        for transparent in [True, False]:
            # Normal usage.
            b = cls(a, 1.5,transparent=transparent)
            b.verify()

            # Negative fade time.
            with pytest.raises(ValueError):
                cls(a, -1)

            # Bogus types as input.
            with pytest.raises(TypeError):
                cls(-1, a)

            # Fade time longer than clip.
            with pytest.raises(ValueError):
                cls(a, 10)

def test_slice_out1():
    # Bad times.
    a = black(640, 480, 30, 3)
    with pytest.raises(TypeError):
        slice_out(0,0,0)
    with pytest.raises(TypeError):
        slice_out(a, a, a)
    with pytest.raises(ValueError):
        slice_out(a, 2, 1)
    with pytest.raises(ValueError):
        slice_out(a, -1, 1)

def test_slice_out2():
    # Bad times.
    a = black(640, 480, 30, 3)
    b = slice_out(a, 1.5, 2.5)
    b.verify()
    assert b.length() == 2

def test_letterbox():
    a = white(640, 480, 30, 3)
    b = letterbox(a, 1000, 1000)
    b.verify()

def test_repeat_frame():
    a = from_file("test_files/bunny.webm", decode_chunk_length=1.0)
    a = slice_clip(a, 0, 1)

    b = repeat_frame(a, 0.2, 5)
    b.verify()
    assert b.length() == 5
    assert b.frame_signature(0) == a.frame_signature(int(0.2*a.frame_rate()))

def test_hold_at_end1():
    # Normal usage.
    a = from_file("test_files/bunny.webm", decode_chunk_length=1.0)
    a = slice_clip(a, 0, 1)

    b = hold_at_end(a, 5)
    b.verify()
    assert b.length() == 5

def test_hold_at_end2():
    # When length is not an exact number of frames.
    a = from_file("test_files/bunny.webm", decode_chunk_length=1.0)
    a = slice_clip(a, 0, 0.98)
    b = hold_at_end(a, 5)
    b.verify(verbose=True)
    assert b.length() == 5

def test_image_glob1():
    a = image_glob("test_files/bunny_frames/*.png", frame_rate=24)
    a.verify()

def test_image_glob2():
    with pytest.raises(FileNotFoundError):
        image_glob("test_files/bunny_frames/*.poo", frame_rate=24)

def test_image_glob3():
    # Make sure we can still find the files if the current directory changes.
    a = image_glob("test_files/bunny_frames/*.png", frame_rate=24)
    a.verify()

    with temporary_current_directory():
        a.verify()

def test_zip_file1():
    a = zip_file("test_files/bunny.zip", frame_rate=15)
    a.verify()

def test_zip_file2():
    with pytest.raises(FileNotFoundError):
        zip_file("test_files/bunny.zap", frame_rate=15)


def test_to_default_metrics():
    a = from_file("test_files/bunny.webm", decode_chunk_length=None)
    a = slice_clip(a, 0, 1.0)

    with pytest.raises(ValueError):
        a.metrics.verify_compatible_with(Clip.default_metrics)

    b = to_default_metrics(a)
    b.verify()
    b.metrics.verify_compatible_with(Clip.default_metrics)

    # Stereo to mono.
    Clip.default_metrics.num_channels = 1
    c = to_default_metrics(a)
    c.verify()
    c.metrics.verify_compatible_with(Clip.default_metrics)

    # Mono to stereo.
    Clip.default_metrics.num_channels = 2
    d = to_default_metrics(c)
    d.verify()
    d.metrics.verify_compatible_with(Clip.default_metrics)

    # Don't know how to deal with 3 channels.
    Clip.default_metrics.num_channels = 3
    with pytest.raises(NotImplementedError):
        to_default_metrics(a)
    Clip.default_metrics.num_channels = 2

def test_timewarp():
    a = white(640, 480, 30, 3)
    b = timewarp(a, 2)
    b.verify()
    assert 2*b.length() == a.length()

def test_pdf_page1():
    a = pdf_page("test_files/snowman.pdf", page_num=1, frame_rate=10, length=3)
    a.verify()

def test_pdf_page2():
    a = pdf_page("test_files/snowman.pdf",
                 page_num=1,
                 frame_rate=10,
                 length=3,
                 size=(101,120))
    a.verify()
    assert a.width() == 101
    assert a.height() == 120

def test_spin():
    a = static_image("test_files/flowers.png", 30, 5)
    b = spin(a, 2)
    b.verify()

def test_vstack():
    a = static_image("test_files/flowers.png", 30, 3)
    b = static_image("test_files/water.png", 30, 5)

    c = vstack(a, b, align=Align.LEFT)
    c.verify()

    d = vstack(a, b, align=Align.RIGHT)
    d.verify()

    e = vstack(a, b, align=Align.CENTER)
    e.verify()

    with pytest.raises(NotImplementedError):
        vstack(a, b, align=Align.TOP)

def test_superimpose_center():
    a = static_image("test_files/flowers.png", 30, 3)
    b = static_image("test_files/water.png", 30, 5)

    c = superimpose_center(a, b, 0)
    c.verify()

def test_loop():
    a = static_image("test_files/flowers.png", 30, 1.2)
    b = spin(a, 1)
    c = loop(b, 10)
    c.verify()
    assert c.length() == 10

def test_ken_burns1():
    # Legit.
    a = static_image("test_files/flowers.png", 30, 10)
    b = ken_burns(clip=a,
                  width=520,
                  height=520,
                  start_top_left=[0,0],
                  start_bottom_right=[100,100],
                  end_top_left=[100,100],
                  end_bottom_right=[250,250])
    b.verify()

def test_ken_burns2():
    # Small distortion: OK.  1.779291553133515 vs 1.7777777777777777
    a = static_image("test_files/flowers.png", 30, 10)
    a2 = scale_to_size(a, width=2945, height=1656)
    b = ken_burns(clip=a2,
                  width=1024,
                  height=576,
                  start_top_left=(63,33),
                  start_bottom_right=(2022,1134),
                  end_top_left=(73,43),
                  end_bottom_right=(2821,1588))
    b.verify()

def test_ken_burns3():
    # Big distortions: Bad.
    a = static_image("test_files/flowers.png", 30, 10)

    with pytest.raises(ValueError):
        ken_burns(clip=a,
                  width=520,
                  height=520,
                  start_top_left=[0,0],
                  start_bottom_right=[100,100],
                  end_top_left=[100,100],
                  end_bottom_right=[250,200])

    with pytest.raises(ValueError):
        ken_burns(clip=a,
                  width=520,
                  height=520,
                  start_top_left=[0,0],
                  start_bottom_right=[100,150],
                  end_top_left=[100,100],
                  end_bottom_right=[200,200])


    with pytest.raises(ValueError):
        ken_burns(clip=a,
                  width=520,
                  height=520,
                  start_top_left=[0,0],
                  start_bottom_right=[100,100],
                  end_top_left=[2000,2000],
                  end_bottom_right=[3000,3000]).verify()


def test_ken_burns4():
    # Grabbing at the exact width or height: OK, because the slice is not
    # inclusive.
    a = black(width=100, height=100, frame_rate=30, length=3)
    b = ken_burns(clip=a,
                  width=50,
                  height=50,
                  start_top_left=(10,10),
                  start_bottom_right=(100,100),
                  end_top_left=(10,10),
                  end_bottom_right=(100,100))
    b.verify()

def test_fade_between():
    a = black(640, 480, 30, 3)
    b = white(640, 480, 30, 3)

    # Normal use.
    c = fade_between(a, b)
    c.verify()

    # Must have same length.
    d = white(640, 480, 30, 4)
    with pytest.raises(ValueError):
        fade_between(a, d)

def test_silence_audio():
    a = from_file("test_files/bunny.webm", decode_chunk_length=5)
    a = slice_clip(a, 0, 5)
    b = silence_audio(a)
    b.verify()

def test_read_image():
    with pytest.raises(FileNotFoundError):
        read_image("xyz.png")

# Grab all of the test source files first.  (...instead of checking within each
# test.)
get_test_files()

# Use a different-from-default directory for the cache, to reduce the pain
# of running these tests (which clear the cache on occasion) when useful
# things are in the default cache.
cache.directory = '/tmp/clipcache-test'
cache.clear()


# If we're run as a script, just execute all of the tests.  Or, if a command
# line argument is given, execute only the tests containing that pattern.
if __name__ == '__main__':  #pragma: no cover
    try:
        pattern = sys.argv[1]
    except IndexError:
        pattern = ""
    for name, thing in list(globals().items()):
        if 'test_' in name and pattern in name:
            print('-'*80)
            print(name)
            print('-'*80)
            thing()
            print()

