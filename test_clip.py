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

    exists = os.path.exists("test_files/ethnocentric_rg.otf")
    exists = exists and os.path.exists("test_files/ethnocentric_rg_it.otf")
    if not exists:
        with urllib.request.urlopen("https://dl.dafont.com/dl/?f=ethnocentric") as u:
            zip_data = u.read()
        file_like_object = io.BytesIO(zip_data)
        with zipfile.ZipFile(file_like_object) as z:
            with open("test_files/ethnocentric_rg.otf", 'wb') as f, \
              z.open("ethnocentric rg.otf") as otf:
                f.write(otf.read())
            with open("test_files/ethnocentric_rg_it.otf", 'wb') as f, \
              z.open("ethnocentric rg it.otf") as otf:
                f.write(otf.read())

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

def test_temporary_current_directory():
    with temporary_current_directory():
        assert glob.glob('*') == []

def test_customprogressbar():
    with custom_progressbar("Testing", 100) as pb:
        pb.update(0)
        for i in range(100):
            pb.update(i)

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

def test_get_font():
    get_font("test_files/ethnocentric_rg.otf", 10)
    get_font("test_files/ethnocentric_rg.otf", 10)
    get_font("test_files/ethnocentric_rg_it.otf", 20)

    with pytest.raises(ValueError):
        get_font("clip3.py", 20)

    with pytest.raises(ValueError):
        get_font("test_files/asdasdasdsad.otf", 20)

def test_save1():
    # Basic case, with video.
    x = solid([0,0,0], 640, 480, 10)
    with temporary_current_directory():
        print(os.getcwd())
        x.save('test.mp4', frame_rate=30, cache_dir=os.getcwd())
        assert os.path.exists('test.mp4')
        x.save('test.mp4', frame_rate=30, cache_dir=os.getcwd())

def test_save2():
    # Pure audio output.
    x = solid([0,0,0], 640, 480, 10)
    with temporary_current_directory():
        x.save('foo.flac', frame_rate=30)
        assert os.path.exists('foo.flac')

def test_save3():
    # With a target filesize.
    a = from_file("test_files/bunny.webm")
    for ts in [5, 10]:
        with temporary_current_directory():
            a.save('small_bunny.mp4',
                   frame_rate=30,
                   target_size=ts,
                   two_pass=True)
            actual_bytes = os.path.getsize("small_bunny.mp4")
            target_bytes = 2**20*ts
            difference = abs(actual_bytes - target_bytes)
            margin = 0.02 * target_bytes
            assert difference < margin, f'{difference} {margin}'


def test_save4():
    # With a target bitrate.
    x = solid([0,0,0], 640, 480, 2)
    with temporary_current_directory():
        x.save('test.mp4', frame_rate=30, bitrate='1024k')
        assert os.path.exists('test.mp4')

def test_save5():
    # With both file size and bitrate.
    x = solid([0,0,0], 640, 480, 2)
    with pytest.raises(ValueError):
        x.save('test.mp4', frame_rate=30, bitrate='1024k', target_size=5)

def test_cache1():
    # Create a new directory when needed.  Remove it when we clear the cache.
    with temporary_current_directory():
        directory = os.path.join(os.getcwd(), 'xyzzy')
        assert not os.path.isdir(directory)
        cache = ClipCache(os.path.join(os.getcwd(), 'xyzzy'))
        cache.scan_directory()
        assert os.path.isdir(directory)
        cache.clear()
        assert not os.path.isdir(directory)

def test_cache2():
    with temporary_current_directory():
        c = ClipCache(directory=os.getcwd())
        c.clear()

        x = solid([0,0,0], 640, 480, 300)
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


def test_metrics_from_ffprobe_output1():
    video_deets = "stream|index=0|codec_name=h264|codec_long_name=H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10|profile=High|codec_type=video|codec_time_base=1/60|codec_tag_string=avc1|codec_tag=0x31637661|width=1024|height=576|coded_width=1024|coded_height=576|has_b_frames=2|sample_aspect_ratio=N/A|display_aspect_ratio=N/A|pix_fmt=yuv420p|level=32|color_range=unknown|color_space=unknown|color_transfer=unknown|color_primaries=unknown|chroma_location=left|field_order=unknown|timecode=N/A|refs=1|is_avc=true|nal_length_size=4|id=N/A|r_frame_rate=30/1|avg_frame_rate=30/1|time_base=1/15360|start_pts=0|start_time=0.000000|duration_ts=1416192|duration=92.200000|bit_rate=1134131|max_bit_rate=N/A|bits_per_raw_sample=8|nb_frames=2766|nb_read_frames=N/A|nb_read_packets=N/A|disposition:default=1|disposition:dub=0|disposition:original=0|disposition:comment=0|disposition:lyrics=0|disposition:karaoke=0|disposition:forced=0|disposition:hearing_impaired=0|disposition:visual_impaired=0|disposition:clean_effects=0|disposition:attached_pic=0|disposition:timed_thumbnails=0|tag:language=und|tag:handler_name=VideoHandler" # pylint: disable=line-too-long
    audio_deets = "stream|index=1|codec_name=aac|codec_long_name=AAC (Advanced Audio Coding)|profile=LC|codec_type=audio|codec_time_base=1/44100|codec_tag_string=mp4a|codec_tag=0x6134706d|sample_fmt=fltp|sample_rate=44100|channels=2|channel_layout=stereo|bits_per_sample=0|id=N/A|r_frame_rate=0/0|avg_frame_rate=0/0|time_base=1/44100|start_pts=0|start_time=0.000000|duration_ts=4066020|duration=92.200000|bit_rate=128751|max_bit_rate=128751|bits_per_raw_sample=N/A|nb_frames=3972|nb_read_frames=N/A|nb_read_packets=N/A|disposition:default=1|disposition:dub=0|disposition:original=0|disposition:comment=0|disposition:lyrics=0|disposition:karaoke=0|disposition:forced=0|disposition:hearing_impaired=0|disposition:visual_impaired=0|disposition:clean_effects=0|disposition:attached_pic=0|disposition:timed_thumbnails=0|tag:language=und|tag:handler_name=SoundHandler" # pylint: disable=line-too-long
    short_audio_deets = "stream|index=1|codec_name=aac|codec_long_name=AAC (Advanced Audio Coding)|profile=LC|codec_type=audio|codec_time_base=1/44100|codec_tag_string=mp4a|codec_tag=0x6134706d|sample_fmt=fltp|sample_rate=44100|channels=2|channel_layout=stereo|bits_per_sample=0|id=N/A|r_frame_rate=0/0|avg_frame_rate=0/0|time_base=1/44100|start_pts=0|start_time=0.000000|duration_ts=4066020|duration=91.000000|bit_rate=128751|max_bit_rate=128751|bits_per_raw_sample=N/A|nb_frames=3972|nb_read_frames=N/A|nb_read_packets=N/A|disposition:default=1|disposition:dub=0|disposition:original=0|disposition:comment=0|disposition:lyrics=0|disposition:karaoke=0|disposition:forced=0|disposition:hearing_impaired=0|disposition:visual_impaired=0|disposition:clean_effects=0|disposition:attached_pic=0|disposition:timed_thumbnails=0|tag:language=und|tag:handler_name=SoundHandler" # pylint: disable=line-too-long
    bogus_deets = "stream|index=1|codec_name=aac|profile=LC|codec_type=trash"

    correct_metrics = Metrics(
      width=1024,
      height=576,
      sample_rate=44100,
      num_channels=2,
      length=92.2
    )
    correct_frame_rate = 30.0

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

    m, fr, _, _ = metrics_from_ffprobe_output(f'{audio_deets}\n{video_deets}', 'test.mp4')
    assert m == correct_metrics
    assert fr == correct_frame_rate

    m, fr, _, _ = metrics_from_ffprobe_output(f'{video_deets}\n{audio_deets}', 'test.mp4')
    assert m == correct_metrics
    assert fr == correct_frame_rate

    m, fr, _, _ = metrics_from_ffprobe_output(f'{video_deets}', 'test.mp4')
    assert m == Metrics(src=correct_metrics,
                        sample_rate=Clip.default_metrics.sample_rate,
                        num_channels=Clip.default_metrics.num_channels)
    assert fr == correct_frame_rate

    m, fr, _, _ = metrics_from_ffprobe_output(f'{audio_deets}', 'test.mp4')
    assert m == Metrics(src=correct_metrics,
                        width=Clip.default_metrics.width,
                        height=Clip.default_metrics.height)
    assert fr is None

def test_metrics_from_ffprobe_output2():
    rotated_video_deets = "stream|index=0|codec_name=h264|codec_long_name=H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10|profile=Baseline|codec_type=video|codec_time_base=18821810/1129461271|codec_tag_string=avc1|codec_tag=0x31637661|width=1600|height=1200|coded_width=1600|coded_height=1200|has_b_frames=0|sample_aspect_ratio=1:1|display_aspect_ratio=4:3|pix_fmt=yuvj420p|level=10|color_range=pc|color_space=smpte170m|color_transfer=smpte170m|color_primaries=bt470bg|chroma_location=left|field_order=unknown|timecode=N/A|refs=1|is_avc=true|nal_length_size=4|id=N/A|r_frame_rate=30/1|avg_frame_rate=1129461271/37643620|time_base=1/90000|start_pts=0|start_time=0.000000|duration_ts=128477601|duration=1427.528900|bit_rate=18000964|max_bit_rate=N/A|bits_per_raw_sample=8|nb_frames=42832|nb_read_frames=N/A|nb_read_packets=N/A|disposition:default=1|disposition:dub=0|disposition:original=0|disposition:comment=0|disposition:lyrics=0|disposition:karaoke=0|disposition:forced=0|disposition:hearing_impaired=0|disposition:visual_impaired=0|disposition:clean_effects=0|disposition:attached_pic=0|disposition:timed_thumbnails=0|tag:rotate=90|tag:creation_time=2020-08-18T15:50:05.000000Z|tag:language=eng|tag:handler_name=VideoHandle" #pylint: disable=line-too-long
    m, _, _, _ = metrics_from_ffprobe_output(f'{rotated_video_deets}', 'test.mp4')
    print(m)


# Grab all of the test source files first.  (...instead of checking within each
# test.)
get_test_files()

# If we're run as a script, just execute all of the tests.  Or, if a
# command line argument is given, execute only the tests containing
# that pattern.
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

