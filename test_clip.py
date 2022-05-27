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

def test_preview():
    x = solid([0,0,0], 640, 480, 5)
    x.preview(30)

def test_save1():
    # Basic case, with video.
    x = solid([0,0,0], 640, 480, 10)
    with temporary_current_directory():
        # Once with an empty cache.
        x.save('test.mp4', frame_rate=30, cache_dir=os.getcwd())
        assert os.path.exists('test.mp4')

        # Again with the cache filled in.
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
    with temporary_current_directory():
        for ts in [5, 10]:
            a.save('small_bunny.mp4',
                   frame_rate=5,
                   target_size=ts,
                   two_pass=True,
                   cache_dir=os.getcwd())
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

def test_get_frame_cached():
    with temporary_current_directory():
        x = solid([0,0,0], 640, 480, 300)
        cache = ClipCache(directory=os.getcwd())
        x.get_frame_cached(cache, 7.5)
        cache.scan_directory()
        x.get_frame_cached(cache, 7.5)


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

def test_from_file1():
    with pytest.raises(FileNotFoundError):
        from_file("test_files/books12312312.mp4")

def test_from_file2():
    a = from_file("test_files/bunny.webm")
    a = slice_clip(a, 0, 1.1)
    a.verify(30)

def test_from_file3():
    b = from_file("test_files/bunny.webm")
    b.verify(30)

    # Again to use the cached dimensions.
    c = from_file("test_files/bunny.webm")
    c.verify(30)

def test_from_file4():
    # For the case with no video.
    d = from_file("test_files/music.mp3")
    assert not d.has_video
    d.verify(30)

def test_from_file5():
    # For the case with no audio.
    e = from_file("test_files/books.mp4")
    e = slice_clip(e, 1.5, 2.5)
    e.verify(10)

def test_from_file6():
    # Suppress audio and suppress video.
    a = from_file("test_files/bunny.webm", suppress_audio=True)
    assert a.has_audio is False

    b = from_file("test_files/bunny.webm", suppress_video=True)
    assert b.has_audio is True

def test_from_file7():
    # Be sure to get boht cache hits and cache misses.
    fname = os.path.join(os.getcwd(), 'test_files/bunny.webm')

    with temporary_current_directory():
        for _ in range(2):
            x = from_file(fname, cache_dir=os.getcwd())
            x.verify(x.frame_rate)

def test_slice_clip():
    a = join(
      solid([0,0,0], 640, 480, 10),
      sine_wave(880, 0.1, 10, 48000, 2)
    )

    with pytest.raises(ValueError):
        slice_clip(a, -1, 1)

    with pytest.raises(ValueError):
        slice_clip(a, 3, 1)

    d = slice_clip(a, 3, 4)
    d.verify(30)

    e = slice_clip(a, 3)
    e.verify(30)

    f = slice_clip(a, end=3)
    f.verify(30)

def asff_helper(fname,
                expected_num_samples,
                expected_num_channels,
                expected_sample_rate):
    """ Run audio_samples_from_file on the given input and sanity check the
    result. """

    cache = ClipCache('/tmp/clipcache/computed')

    # Grab the audio samples.
    s = audio_samples_from_file(fname,
                                cache,
                                expected_num_samples=expected_num_samples,
                                expected_num_channels=expected_num_channels,
                                expected_sample_rate=expected_sample_rate)

    # Make sure we got the right sort of matrix back.
    assert s.shape == (expected_num_samples, expected_num_channels)
    assert s.dtype == np.float64

    # If they're all 0, something is wrong.  This was
    # failing for a while when we mistakenly tried to put
    # float64 data into a uint numpy array.
    assert s.any()

def test_audio_samples_from_file1():
    with pytest.raises(FFMPEGException):
        # No audio track.
        asff_helper(
          "test_files/books.mp4",
          expected_num_samples=0,
          expected_num_channels=1,
          expected_sample_rate=0
        )

def test_audio_samples_from_file2():
    with pytest.raises(ValueError):
        # Wrong sample rate.
        asff_helper(
          "test_files/music.mp3",
          expected_num_samples=3335168,
          expected_num_channels=2,
          expected_sample_rate=48000
        )

def test_audio_samples_from_file3():
    with pytest.raises(ValueError):
        # Wrong number of channels
        asff_helper(
          "test_files/music.mp3",
          expected_num_samples=3335168,
          expected_num_channels=1,
          expected_sample_rate=44100
        )

def test_audio_samples_from_file4():
    with pytest.raises(ValueError):
        # Wrong length.
        asff_helper(
          "test_files/music.mp3",
          expected_num_samples=4335170,
          expected_num_channels=2,
          expected_sample_rate=44100
        )

def test_audio_samples_from_file5():
    # Slightly too long.
    asff_helper(
      "test_files/music.mp3",
      expected_num_samples=3337343,
      expected_num_channels=2,
      expected_sample_rate=44100
    )

def test_audio_samples_from_file6():
    # Slightly too short.
    asff_helper(
      "test_files/music.mp3",
      expected_num_samples=3337345,
      expected_num_channels=2,
      expected_sample_rate=44100
    )

def test_audio_samples_from_file7():
    # All good.
    asff_helper(
      "test_files/music.mp3",
      expected_num_samples=3337344,
      expected_num_channels=2,
      expected_sample_rate=44100
    )

def test_audio_samples_from_file8():
    # Ensure a cache miss and a cache hit.
    fname = os.path.join(os.getcwd(), 'test_files/music.mp3')

    with temporary_current_directory():
        cache = ClipCache(os.getcwd())
        for _ in range(2):
            audio_samples_from_file(fname,
                                    cache,
                                    expected_num_samples=3337344,
                                    expected_num_channels=2,
                                    expected_sample_rate=44100)

def test_alpha_blend():
    f0 = cv2.imread("test_files/flowers.png", cv2.IMREAD_UNCHANGED)
    f1 = np.zeros(shape=f0.shape, dtype=np.uint8)
    f2 = alpha_blend(f0, f1)
    cv2.imwrite('test_files/blended.png', f2)

    f0 = cv2.imread("test_files/water.png", cv2.IMREAD_UNCHANGED)
    f0 = f0[0:439,:,:]
    f1 = cv2.imread("test_files/flowers.png", cv2.IMREAD_UNCHANGED)
    f2 = alpha_blend(f0, f1)

def test_composite1():
    # Automatically compute the height.
    x = solid([0,0,0], 640, 480, 5)
    y = solid([0,0,0], 640, 481, 5)
    z = composite(
      Element(x, 0, [0, 0]),
      Element(y, 6, [0, 0])
    )
    z.verify(30)
    assert z.height() == 481

def test_composite2():
    # Can't start before 0.
    x = solid([0,0,0], 640, 480, 5)
    y = solid([0,0,0], 640, 480, 5)

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
    x = solid([0,0,0], 640, 480, 5)
    y = solid([0,0,0], 640, 480, 5)
    z = composite(
      Element(x, 0, [0, 0]),
      Element(y, 6, [0, 0])
    )
    assert z.length() == 11
    z.verify(30)

def test_composite6():
    # Clipping above, below, left, and right.
    x = static_image("test_files/flowers.png", 10)
    x = scale_by_factor(x, 0.4)

    z = composite(
      Element(x, 0, [-100, -100], VideoMode.REPLACE),
      Element(x, 0, [540, 380], VideoMode.REPLACE),
      width=640,
      height=480,
      length=5
    )
    z.verify(30)

def test_composite7():
    # Totally off-screen.
    x = static_image("test_files/flowers.png", 10)
    x = scale_by_factor(x, 0.4)

    z = composite(
      Element(x, 0, [-1000, -100], VideoMode.REPLACE),
      Element(x, 0, [1540, 380], VideoMode.REPLACE),
      width=640,
      height=480,
      length=5
    )
    z.verify(30)


def test_composite8():
    # Alpha blending.
    x = static_image("test_files/flowers.png", 5000)
    x = scale_by_factor(x, 0.4)

    z = composite(
      Element(x, 0, [50, 50], video_mode=VideoMode.BLEND),
      Element(x, 0, [250, 150], video_mode=VideoMode.BLEND),
      width=640,
      height=480,
      length=1
    )
    z.verify(30)

def test_composite9():
    # Bad inputs.
    x = static_image("test_files/flowers.png", 5000)
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
    x = static_image("test_files/flowers.png", 5)
    x = scale_by_factor(x, 0.4)

    def pos1(t):
        return [t-100,2*t-100]
    def pos2(t):
        return [480-t,2*t-100]

    z = composite(
      Element(x, 0, pos1, video_mode=VideoMode.BLEND),
      Element(x, 0, pos2, video_mode=VideoMode.BLEND),
      length=5
    )
    z.verify(30)

def test_composite11():
    # Ignored video should not impact the frame signatures.
    a = static_image("test_files/flowers.png", 5)

    b = composite(
      Element(a, 0, [0,0], video_mode=VideoMode.BLEND),
    )

    c = composite(
      Element(a, 0, [0,0], video_mode=VideoMode.BLEND),
      Element(a, 0, [10,10], video_mode=VideoMode.IGNORE),
    )

    assert b.frame_signature(0) == c.frame_signature(0)

def test_join1():
    # Normal case.
    x = sine_wave(440, 0.25, 3, 48000, 2)
    y = solid([0,255,0], 640, 480, 5)
    z = join(y, x)
    z.verify(30)
    assert y.length() == 5

def test_join2():
    # Detect that audio and video are swapped.
    x = sine_wave(440, 0.25, 3, 48000, 2)
    y = solid([0,255,0], 640, 480, 5)
    with pytest.raises(AssertionError):
        join(x, y)

def test_filter_frames1():
    a = black(640, 480, 3)

    b = filter_frames(a, lambda x: x)
    assert not b.depends_on_time
    b.verify(30)

    c = filter_frames(a, lambda x: x, name='identity')
    c.verify(30)

    d = filter_frames(a, lambda x: x, size='same')
    d.verify(30)

    e = filter_frames(a, lambda x: x, size=(a.width(), a.height()))
    e.verify(30)

    # Nonsense size
    with pytest.raises(ValueError):
        filter_frames(a, lambda x: x, size='sooom')

    # Wrong size
    f = filter_frames(a, lambda x: x, size=(10, 10))
    with pytest.raises(ValueError):
        f.verify(30)

    # Signatures match?
    g = filter_frames(a, lambda x: x)
    h = filter_frames(a, lambda x: x)
    i = filter_frames(a, lambda x: x+1)
    assert g.sig == h.sig
    assert h.sig != i.sig

def test_filter_frames2():
    # Two-parameter filter version.
    a = black(640, 480, 3)
    b = filter_frames(a, lambda x, i: x)
    b.verify(30)

def test_filter_frames3():
    # A bogus filter.
    a = black(640, 480, 3)
    with pytest.raises(TypeError):
        filter_frames(a, lambda x, y, z: None)

def test_scale_to_size():
    a = black(640, 480, 3)
    b = scale_to_size(a, 100, 200)
    b.verify(30)
    assert b.width() == 100
    assert b.height() == 200

def test_scale_by_factor():
    a = black(100, 200, 3)
    b = scale_by_factor(a, 0.1)
    b.verify(30)
    assert b.width() == 10
    assert b.height() == 20

def test_scale_to_fit():
    a = black(100, 100, 3)
    b = scale_to_fit(a, 50, 100)
    b.verify(30)
    assert abs(b.width()/b.height() - 1.0)  < 1e-10

    c = scale_to_fit(a, 100, 50)
    c.verify(30)
    assert abs(b.width()/b.height() - 1.0)  < 1e-10

def test_static_frame1():
    # Legit usage: An RGBA image.
    a = static_image("test_files/water.png", 10)
    a.verify(30)

def test_static_frame2():
    # Legit usage: An RGB image.
    b = static_image("test_files/brian.jpg", 10)
    b.verify(30)


def test_static_frame3():
    # Wrong type
    with pytest.raises(TypeError):
        static_frame("not a frame", "name", 10)

def test_static_frame4():
    # Wrong shape
    with pytest.raises(ValueError):
        static_frame(np.zeros([100, 100]), "name", 10)

def test_static_frame5():
    # Wrong number of channels
    with pytest.raises(ValueError):
        static_frame(np.zeros([100, 100, 3]), "name", 10)

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

    a = static_frame(black_frame, "blackness", 3)
    b = static_frame(black_frame2, "blackness", 3)
    c = static_frame(white_frame, "whiteness", 3)
    d = static_frame(mostly_black_frame, "mostly black", 3)

    assert a.sig == b.sig
    assert a.sig != c.sig
    assert a.sig != d.sig

def test_chain():
    a = black(640, 480, 3)
    b = white(640, 480, 3)
    c = solid([255,0,0], 640, 480, 3)

    d = chain(a, [b, c])
    assert d.length() == a.length() + b.length() + c.length()
    d.verify(30)

    e = chain(a, [b, c], fade_time=2)
    assert e.length() == a.length() + b.length() + c.length() - 4
    e.verify(30)

    with pytest.raises(ValueError):
        chain()

    with pytest.raises(ValueError):
        chain(fade_time=3)

def test_fades():
    a = white(640, 480, 3)

    for cls in [fade_in, fade_out]:
        for transparent in [True, False]:
            # Normal usage.  Very high frame rate, to cover the case where
            # some frames are unchanged.
            b = cls(a, 1.5,transparent=transparent)
            b.verify(300)

            # Negative fade time.
            with pytest.raises(ValueError):
                cls(a, -1)

            # Bogus types as input.
            with pytest.raises(TypeError):
                cls(-1, a)

            # Fade time longer than clip.
            with pytest.raises(ValueError):
                cls(a, 10)

def test_mono_to_stereo():
    a = sine_wave(880, 0.1, 10, 48000, 1)
    b = mono_to_stereo(a)
    b.verify(30)
    assert b.num_channels() == 2

    a = sine_wave(880, 0.1, 10, 48000, 2)
    with pytest.raises(ValueError):
        # Not a mono source.
        b = mono_to_stereo(a)

def test_stereo_to_mono():
    a = sine_wave(880, 0.1, 10, 48000, 2)
    b = stereo_to_mono(a)
    b.verify(30)
    assert b.num_channels() == 1

    a = sine_wave(880, 0.1, 10, 48000, 1)
    with pytest.raises(ValueError):
        # Not a stereo source.
        b = stereo_to_mono(a)

def test_reverse():
    a = join(
      solid([0,0,0], 640, 480, 1),
      sine_wave(880, 0.1, 1, 48000, 2)
    )
    b = join(
      solid([255,0,0], 640, 480, 1),
      sine_wave(440, 0.1, 1, 48000, 2)
    )
    c = chain(a,b)
    d = reverse(c)
    d.verify(30)

    f1 = c.get_frame(5)
    f2 = d.get_frame(d.length()-5)
    assert np.array_equal(f1, f2)

    s1 = c.get_samples()
    s2 = d.get_samples()

    assert np.array_equal(s1[5], s2[s2.shape[0]-6])

def test_volume():
    a = sine_wave(880, 0.1, 10, 48000, 2)
    b = scale_volume(a, 0.1)
    b.verify(30)

    with pytest.raises(ValueError):
        scale_volume(a, -10)

    with pytest.raises(TypeError):
        scale_volume(a, "tim")

def test_crop():
    a = join(
      solid([0,0,0], 640, 480, 10),
      sine_wave(880, 0.1, 10, 48000, 2)
    )

    # Typical usage.
    b = crop(a, [10, 10], [100, 100])
    b.verify(30)
    assert b.width() == 90
    assert b.height() == 90


    # Zero is okay.
    c = crop(a, [0, 0], [100, 100])
    c.verify(30)

    with pytest.raises(ValueError):
        crop(a, [-1, 10], [100, 100])

    with pytest.raises(ValueError):
        crop(a, [100, 100], [10, 10])

    with pytest.raises(ValueError):
        crop(a, [10, 10], [100, 10000])

def test_draw_text():
    font = "test_files/ethnocentric_rg_it.otf"
    x = draw_text("Hello!", font, font_size=200, color=[255,0,255], length=5)
    x.verify(10)

def test_to_monochrome():
    a = black(640, 480, 3)
    b = to_monochrome(a)
    b.verify(30)

def test_resample1():
    # Basic case.
    length = 5
    a = from_file("test_files/bunny.webm")
    a = slice_clip(a, 0, length)

    sr = 48000
    l = 2*length
    b = resample(a, sample_rate=sr, length=l)
    assert b.sample_rate() == sr
    assert b.length() == l
    b.verify(29)

def test_resample2():
    # Cover all of the default-parameter branches.
    length = 5
    a = from_file("test_files/bunny.webm")
    a = slice_clip(a, 0, length)

    b = resample(a)
    b.verify(30)

def test_slice_out1():
    # Bad times.
    a = black(640, 480, 3)
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
    a = black(640, 480, 3)
    b = slice_out(a, 1.5, 2.5)
    b.verify(30)
    assert b.length() == 2

def test_letterbox():
    a = white(640, 480, 3)
    b = letterbox(a, 1000, 1000)
    b.verify(30)

def test_repeat_frame():
    x = from_file("test_files/bunny.webm")
    a = slice_clip(x, 0, 1)

    when = 0.2

    b = repeat_frame(a, when, 5)
    b.verify(x.frame_rate, verbose=False)
    assert b.length() == 5
    assert b.frame_signature(0) == a.frame_signature(when)

def test_hold_at_end1():
    # Normal usage.
    x = from_file("test_files/bunny.webm")
    a = slice_clip(x, 0, 1)

    b = hold_at_end(a, 5)
    b.verify(x.frame_rate)
    assert b.length() == 5

def test_hold_at_end2():
    # When length is not an exact number of frames.
    x = from_file("test_files/bunny.webm")
    a = slice_clip(x, 0, 0.98)
    b = hold_at_end(a, 5)
    b.verify(x.frame_rate, verbose=True)
    assert b.length() == 5

def test_image_glob1():
    # Normal usage.
    a = image_glob("test_files/bunny_frames/*.png", frame_rate=24)
    a.verify(24)

def test_image_glob2():
    # Bad pattern.
    with pytest.raises(FileNotFoundError):
        image_glob("test_files/bunny_frames/*.poo", frame_rate=24)

def test_image_glob3():
    # Make sure we can still find the files if the current directory changes.
    a = image_glob("test_files/bunny_frames/*.png", frame_rate=24)
    a.verify(24)

    with temporary_current_directory():
        a.verify(24)

def test_image_glob4():
    # Provide length instead of frame rate.
    a = image_glob("test_files/bunny_frames/*.png", length=315)
    assert a.frame_rate == 10

def test_image_glob5():
    # Bad args.
    with pytest.raises(ValueError):
        image_glob("test_files/bunny_frames/*.png", length=1, frame_rate=1)
    with pytest.raises(ValueError):
        image_glob("test_files/bunny_frames/*.png")

def test_zip_file1():
    a = zip_file("test_files/bunny.zip", frame_rate=15)
    a.verify(30)

def test_zip_file2():
    with pytest.raises(FileNotFoundError):
        zip_file("test_files/bunny.zap", frame_rate=15)

def test_to_default_metrics():
    a = from_file("test_files/bunny.webm")
    a = slice_clip(a, 0, 1.0)

    with pytest.raises(ValueError):
        a.metrics.verify_compatible_with(Clip.default_metrics)

    b = to_default_metrics(a)
    b.verify(30)
    b.metrics.verify_compatible_with(Clip.default_metrics)

    # Stereo to mono.
    Clip.default_metrics.num_channels = 1
    c = to_default_metrics(a)
    c.verify(30)
    c.metrics.verify_compatible_with(Clip.default_metrics)

    # Mono to stereo.
    Clip.default_metrics.num_channels = 2
    d = to_default_metrics(c)
    d.verify(30)
    d.metrics.verify_compatible_with(Clip.default_metrics)

    # Don't know how to deal with 3 channels.
    Clip.default_metrics.num_channels = 3
    with pytest.raises(NotImplementedError):
        to_default_metrics(a)
    Clip.default_metrics.num_channels = 2

def test_timewarp():
    a = white(640, 480, 3)
    b = timewarp(a, 2)
    b.verify(30)
    assert 2*b.length() == a.length()

def test_pdf_page1():
    a = pdf_page("test_files/snowman.pdf",
                 page_num=1,
                 length=3)
    a.verify(30)

def test_pdf_page2():
    a = pdf_page("test_files/snowman.pdf",
                 page_num=1,
                 length=3,
                 size=(101,120))
    a.verify(10)
    assert a.width() == 101
    assert a.height() == 120

def test_spin():
    a = static_image("test_files/flowers.png", 5)
    b = spin(a, 2)
    b.verify(30)

def test_vstack():
    a = static_image("test_files/flowers.png", 3)
    b = static_image("test_files/water.png", 5)

    c = vstack(a, b, align=Align.LEFT)
    c.verify(30)

    d = vstack(a, b, align=Align.RIGHT)
    d.verify(30)

    e = vstack(a, b, align=Align.CENTER)
    e.verify(30)

    with pytest.raises(NotImplementedError):
        vstack(a, b, align=Align.TOP)

def test_hstack():
    a = static_image("test_files/flowers.png", 3)
    b = static_image("test_files/water.png", 5)

    c = hstack(a, b, align=Align.TOP)
    c.verify(30)

    d = hstack(a, b, align=Align.BOTTOM)
    d.verify(30)

    e = hstack(a, b, align=Align.CENTER)
    e.verify(30)

    with pytest.raises(NotImplementedError):
        hstack(a, b, align=Align.LEFT)

def test_stack_clips():
    a = static_image("test_files/flowers.png", 3)
    b = static_image("test_files/water.png", 5)

    # Integer for spacing in the list
    c = stack_clips(a, 10, b, align=Align.LEFT, vert=True, name='vstack')
    c.verify(30)

    # Junk in the list
    with pytest.raises(TypeError):
        stack_clips(a, 1.2, b, align=Align.LEFT, vert=True, name='vstack')




# Grab all of the test source files first.  (...instead of checking within
# each test.)
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

