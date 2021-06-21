#!/usr/bin/env python3
# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=wildcard-import

import glob
import shutil
import urllib.request

import cv2
import pytest

from clip3 import *


def check_clip(clip):
    """ Fully realize a clip, ensuring that no exception occur and that the
    right sizes of video frames and audio samples are returned. """
    for i in range(clip.num_frames()):
        clip.frame_signature(i)
        frame = clip.get_frame(i)
        assert frame.shape == (clip.height(), clip.width(), 4)

    samples = clip.get_samples()
    assert samples.shape == (clip.num_samples(), clip.num_channels())

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
    x = solid([0,0,0], 640, 480, 30, 300)
    check_clip(x)

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
    with pytest.raises(FFMPEGException):
        ffmpeg('-i /dev/zero', '/dev/null')

    with pytest.raises(FFMPEGException):
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

def test_temporal_composite():
    x = solid([0,0,0], 640, 480, 30, 5)
    y = solid([0,0,0], 640, 481, 30, 5)

    with pytest.raises(ValueError):
        # Heights do not match.
        z = temporal_composite(
          TCE(x, 0),
          TCE(y, 6)
        )

    x = solid([0,0,0], 640, 480, 30, 5)
    y = solid([0,0,0], 640, 480, 30, 5)

    with pytest.raises(ValueError):
        # Can't start before 0.
        z = temporal_composite(
          TCE(x, -1),
          TCE(y, 6)
        )

    x = sine_wave(880, 0.1, 5, 48000, 2)
    y = sine_wave(880, 0.1, 5, 48001, 2)

    with pytest.raises(ValueError):
        # Sample rates don't match.
        z = temporal_composite(
          TCE(x, 0),
          TCE(y, 5)
        )

    x = solid([0,0,0], 640, 480, 30, 5)
    y = solid([0,0,0], 640, 480, 30, 5)
    z = temporal_composite(
      TCE(x, 0),
      TCE(y, 6)
    )

    assert z.length() == 11

    check_clip(z)

    cache.clear()
    z.preview()
    z.get_samples()


def test_sine_wave():
    x = sine_wave(880, 0.1, 5, 48000, 2)
    check_clip(x)

def test_join():
    x = sine_wave(440, 0.25, 5, 48000, 2)
    y = solid([0,255,0], 640, 480, 30, 5)
    z = join(y, x)
    check_clip(z)

    with pytest.raises(AssertionError):
        join(x, y)


def test_chain_and_fade_chain():
    a = black(640, 480, 30, 3)
    b = white(640, 480, 30, 3)
    c = solid([255,0,0], 640, 480, 30, 3)

    d = chain(a, [b, c])
    assert d.length() == a.length() + b.length() + c.length()
    check_clip(d)

    e = fade_chain(2, a, [b, c])
    assert e.length() == a.length() + b.length() + c.length() - 4
    check_clip(e)

    with pytest.raises(ValueError):
        chain()

    with pytest.raises(ValueError):
        fade_chain(3)


def test_black_and_white():
    check_clip(black(640, 480, 30, 300))
    check_clip(white(640, 480, 30, 300))

def test_mutator():
    a = black(640, 480, 30, 5)
    b = MutatorClip(a)
    check_clip(b)

def test_scale_alpha():
    a = black(640, 480, 30, 5)
    b = scale_alpha(a, 0.5)
    check_clip(b)


def test_preview():
    cache.clear()
    x = solid([0,0,0], 640, 480, 30, 5)
    x.preview()

def test_metrics_from_ffprobe_output():
    video_deets = "stream|index=0|codec_name=h264|codec_long_name=H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10|profile=High|codec_type=video|codec_time_base=1/60|codec_tag_string=avc1|codec_tag=0x31637661|width=1024|height=576|coded_width=1024|coded_height=576|has_b_frames=2|sample_aspect_ratio=N/A|display_aspect_ratio=N/A|pix_fmt=yuv420p|level=32|color_range=unknown|color_space=unknown|color_transfer=unknown|color_primaries=unknown|chroma_location=left|field_order=unknown|timecode=N/A|refs=1|is_avc=true|nal_length_size=4|id=N/A|r_frame_rate=30/1|avg_frame_rate=30/1|time_base=1/15360|start_pts=0|start_time=0.000000|duration_ts=1416192|duration=92.200000|bit_rate=1134131|max_bit_rate=N/A|bits_per_raw_sample=8|nb_frames=2766|nb_read_frames=N/A|nb_read_packets=N/A|disposition:default=1|disposition:dub=0|disposition:original=0|disposition:comment=0|disposition:lyrics=0|disposition:karaoke=0|disposition:forced=0|disposition:hearing_impaired=0|disposition:visual_impaired=0|disposition:clean_effects=0|disposition:attached_pic=0|disposition:timed_thumbnails=0|tag:language=und|tag:handler_name=VideoHandler" # pylint: disable=line-too-long
    audio_deets = "stream|index=1|codec_name=aac|codec_long_name=AAC (Advanced Audio Coding)|profile=LC|codec_type=audio|codec_time_base=1/44100|codec_tag_string=mp4a|codec_tag=0x6134706d|sample_fmt=fltp|sample_rate=44100|channels=2|channel_layout=stereo|bits_per_sample=0|id=N/A|r_frame_rate=0/0|avg_frame_rate=0/0|time_base=1/44100|start_pts=0|start_time=0.000000|duration_ts=4066020|duration=92.200000|bit_rate=128751|max_bit_rate=128751|bits_per_raw_sample=N/A|nb_frames=3972|nb_read_frames=N/A|nb_read_packets=N/A|disposition:default=1|disposition:dub=0|disposition:original=0|disposition:comment=0|disposition:lyrics=0|disposition:karaoke=0|disposition:forced=0|disposition:hearing_impaired=0|disposition:visual_impaired=0|disposition:clean_effects=0|disposition:attached_pic=0|disposition:timed_thumbnails=0|tag:language=und|tag:handler_name=SoundHandler" # pylint: disable=line-too-long
    short_audio_deets = "stream|index=1|codec_name=aac|codec_long_name=AAC (Advanced Audio Coding)|profile=LC|codec_type=audio|codec_time_base=1/44100|codec_tag_string=mp4a|codec_tag=0x6134706d|sample_fmt=fltp|sample_rate=44100|channels=2|channel_layout=stereo|bits_per_sample=0|id=N/A|r_frame_rate=0/0|avg_frame_rate=0/0|time_base=1/44100|start_pts=0|start_time=0.000000|duration_ts=4066020|duration=92.000000|bit_rate=128751|max_bit_rate=128751|bits_per_raw_sample=N/A|nb_frames=3972|nb_read_frames=N/A|nb_read_packets=N/A|disposition:default=1|disposition:dub=0|disposition:original=0|disposition:comment=0|disposition:lyrics=0|disposition:karaoke=0|disposition:forced=0|disposition:hearing_impaired=0|disposition:visual_impaired=0|disposition:clean_effects=0|disposition:attached_pic=0|disposition:timed_thumbnails=0|tag:language=und|tag:handler_name=SoundHandler" # pylint: disable=line-too-long
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

    m = metrics_from_ffprobe_output(f'{audio_deets}\n{video_deets}', 'test.mp4')
    assert m == correct_metrics

    m = metrics_from_ffprobe_output(f'{video_deets}\n{audio_deets}', 'test.mp4')
    assert m == correct_metrics

    m = metrics_from_ffprobe_output(f'{video_deets}', 'test.mp4')
    assert m == Metrics(
      src=correct_metrics,
      sample_rate=default_metrics.sample_rate,
      num_channels=default_metrics.num_channels
    )

    m = metrics_from_ffprobe_output(f'{audio_deets}', 'test.mp4')
    assert m == Metrics(
      src=correct_metrics,
      width=default_metrics.width,
      height=default_metrics.height,
      frame_rate=default_metrics.frame_rate,
    )


def test_from_file():
    if not os.path.exists("books.mp4"):  # pragma: no cover
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve("https://www.pexels.com/video/5224014/download", "books.mp4")

    with pytest.raises(FileNotFoundError):
        from_file("books12312312.mp4")

    from_file("books.mp4")
    # check_clip(a)

    from_file("books.mp4", forced_length=30)


# If we're run as a script, just execute all of the tests.
if __name__ == '__main__':  #pragma: no cover
    for name, thing in list(globals().items()):
        if 'test_' in name:
            thing()

