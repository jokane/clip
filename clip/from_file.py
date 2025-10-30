""" A class for reading clips from files in various formats. """

# pylint: disable=wildcard-import

import glob
import itertools
import os
import re
import subprocess
import time
import warnings

import cv2
import numpy as np
import soundfile

from .audio import patch_audio_length
from .base import Clip, FiniteIndexed
from .cache import ClipCache
from .util import read_image
from .ffmpeg import ffmpeg
from .metrics import Metrics
from .validate import *
from .util import temporary_current_directory, parse_hms_to_seconds

def get_duration_from_ffprobe_stream(stream, fmt):
    """Determine the duration of a stream found by `ffprobe`, using `fmt` if
    available as the container format output.

    :param stream: A dictionary built from the key-value pairs in an `ffprobe`
            stream.
    :param fmt: A dictionary built from the key-value pairs in an `ffprobe`
            format.
    :return: A float number of seconds of duration for that stream.

    Raises `ValueError` if no duration is found.

    """

    # Plan A: Duration given directly as a number of seconds.
    if 'duration' in stream and is_float(stream['duration']):
        return float(stream['duration'])

    # Plan B: Duration given in a tag as HH:MM:SS.XXX.
    tags = ['tag:DURATION',
            'tag:DURATION-eng']

    for tag in tags:
        if tag in stream:
            if match := re.match(r"(\d\d):(\d\d):([0-9\.]+)", stream[tag]):
                hours = float(match.group(1))
                mins = float(match.group(2))
                secs = float(match.group(3))
                return secs + 60*mins + 60*60*hours

    # Plan C: Duration given in the format entry instead.
    if fmt and 'duration' in fmt and is_float(fmt['duration']):
        return float(fmt['duration'])

    raise ValueError(f"Could not find a duration in ffprobe stream. {stream}")

def get_framerate_from_ffprobe_stream(stream):
    """Determine the frame rate for a video stream found by `ffprobe`.

    :param stream: A dictionary built from the key-value pairs in an `ffprobe`
            stream.
    :return: A float number of frames per second for that stream.

    Raises `ValueError` if no frame rate is found.

    """

    # Plan A: Most streams seem to have an avg_frame_rate.
    if 'avg_frame_rate' in stream and stream['avg_frame_rate'] != '0/0':
        return eval(stream['avg_frame_rate'])

    # Plan B: If the avg_frame_rate is missing or nonsense, try r_frame_rate instead.
    if 'r_frame_rate' in stream and stream['r_frame_rate'] != '0/0':
        return eval(stream['r_frame_rate'])

    raise ValueError(f"Could not find a frame rate in ffprobe stream. {stream}")

def metrics_and_frame_rate_from_stream_dicts(streams, filename):
    """Given a dict containing the audio, video, and subtitles streams of
    a clip, return the appropriate :class:`Metrics` object, the float frame rate, and
    booleans telling whether video, audio, and subtitles exist.

    :param streams: A dictionary with
            `"audio"`, or `"subtitle"`.  Each value a dictionary built from the
            key-value pairs in an `ffprobe`

    :param filename: The name of the file described by `streams`.  Used only
            for error messages.

    """
    video_stream = streams['video']
    audio_stream = streams['audio']
    subtitle_stream = streams['subtitle']
    fmt = streams['format']
    has_subtitles = subtitle_stream is not None

    # Some videos, especially from mobile phones, contain metadata asking for a
    # rotation.  We'll generally not try to deal with that here ---better,
    # perhaps, to let the user flip or rotate as needed later--- but
    # landscape/portait differences are important because they affect the width
    # and height, and are respected by ffmpeg when the frames are extracted.
    # Thus, we need to apply that change here to prevent mismatched frame sizes
    # down the line.
    if (video_stream and 'tag:rotate' in video_stream
          and video_stream['tag:rotate'] in ['-90','90']):
        video_stream['width'],video_stream['height'] = video_stream['height'],video_stream['width']

    if video_stream and audio_stream:
        vlen = get_duration_from_ffprobe_stream(video_stream, fmt)
        fr = get_framerate_from_ffprobe_stream(video_stream)
        alen = get_duration_from_ffprobe_stream(audio_stream, fmt)

        if abs(vlen - alen) > 0.5:
            raise ValueError(f"In {filename}, video length ({vlen}) and audio length ({alen}) "
              "do not match. Perhaps load video and audio separately?")

        return Metrics(width = eval(video_stream['width']),
                       height = eval(video_stream['height']),
                       sample_rate = eval(audio_stream['sample_rate']),
                       num_channels = eval(audio_stream['channels']),
                       length = min(vlen, alen)), \
               fr, True, True, has_subtitles
    elif video_stream:
        vlen = get_duration_from_ffprobe_stream(video_stream, fmt)
        fr = get_framerate_from_ffprobe_stream(video_stream)

        return Metrics(src = Clip.default_metrics,
                       width = eval(video_stream['width']),
                       height = eval(video_stream['height']),
                       length = vlen), \
               fr, True, False, has_subtitles
    elif audio_stream:
        alen = get_duration_from_ffprobe_stream(audio_stream, fmt)
        return Metrics(src = Clip.default_metrics,
                       sample_rate = eval(audio_stream['sample_rate']),
                       num_channels = eval(audio_stream['channels']),
                       length = alen), \
               None, False, True, has_subtitles
    else:
        # Should be impossible to get here, but just in case...
        raise ValueError(f"File {filename} contains neither audio nor video.") # pragma: no cover

def metrics_from_ffprobe_output(ffprobe_output, filename, suppress=None):
    """Sift through output from `ffprobe` and trying to make a `Metrics` from it.

    :param ffprobe_output: A string containing output from `ffprobe`.
    :param supress: A list containing some (possibly empty) subset of
            `"video"`, `"audio"`, and `"subtitle"`.  Streams of those types
            will be ignored.

    :return: A :class:`Metrics` object based on that data, or raise an
            exception if something strange is in there.

    The output should specifically be from::

        ffprobe -of compact -show_entries stream

    Used indirectly by :class:`from_file`.
    """

    stream_lists = {'video': [],
                    'audio': [],
                    'subtitle': [],
                    'format': []}

    if suppress is None:
        suppress = ['data']

    for line in ffprobe_output.strip().split('\n'):
        # Each line is a pipe-separated list of key value pairs.  Massage that
        # into a dictionary.
        stream = {}
        fields = line.split('|')

        # Sanity check.
        if fields[0] not in ['stream', 'format']: continue

        # Convert into a dictionary.
        for pair in fields[1:]:
            key, val = pair.split('=', 1)
            if key in stream:
                raise ValueError(f'Stream has multiple values for key {key}: {stream[key]} val')
            stream[key] = val

        # Handle streams and formats differently.
        if fields[0] == 'stream':
            # What kind of stream is this?
            t = stream['codec_type']

            # A special case: Image-based subtitles.  Can't do much with those.
            if t == 'subtitle' and stream['codec_name'] == 'dvd_subtitle':
                warnings.warn(f'Ignoring image-based subtitles in {filename}.')
                continue

            # Store the stream data based on what type it is.  Ignore streams that
            # we are suppressing.  Complain about streams that we are neither
            # expecting nor suppressing.
            if t in suppress:
                continue
            if t in stream_lists:
                stream_lists[t].append(stream)
            else:
                raise ValueError(f"Don't know what to do with {filename}, "
                  f"which has an unknown stream of type {stream['codec_type']}.")
        else:  # fields[0] == 'format':
            stream_lists['format'].append(stream)

    streams_by_type = {}
    for t, streams in stream_lists.items():
        if len(streams)==0:
            streams_by_type[t] = None
        elif len(streams)==1:
            streams_by_type[t] = streams[0]
        else:
            warnings.warn(f'In {filename} there are {len(streams)} {t} streams.  \
                    Using the first one.')
            streams_by_type[t] = streams[0]

    return metrics_and_frame_rate_from_stream_dicts(streams_by_type, filename)

def audio_samples_from_file(filename, cache, expected_sample_rate, expected_num_channels,
                            expected_num_samples):
    """Extract audio data from a file, which may be either a pure audio format
    or a video file containing an audio stream.

    :param filename: The name of the file to read.
    :param cache: A :class:`ClipCache` that might have the audio we want, or
            into which it can be stored.
    :param expected_sample_rate: The sample rate we expect to see in the audio,
            in samples per second.
    :param expected_num_channels: The number of channels we expect to see in
            the audio, usually either `1` or `2`.
    :param expected_num_samples: The number of audio samples we expect to see
            in each channel.

    Raise an exception if the audio that turns up does not match the expected
    sample rate, number of channels, and approximate ---that is, within about 1
    second--- number of samples.
    """

    # Grab the file's extension.
    ext = os.path.splitext(filename)[1].lower()

    # Is it in a format we know how to read directly?  If so, read it and
    # declare victory.
    direct_formats = list(map(lambda x: "." + x.lower(),
      soundfile.available_formats().keys()))
    if ext in direct_formats:
        print("Reading audio from", filename)

        # Acquire the data from the file.
        data, sample_rate = soundfile.read(filename, always_2d=True)

        # Complain if the sample rates or numbers of channels don't match.
        if sample_rate != expected_sample_rate:
            raise ValueError(f"From {filename}, expected sample rate {expected_sample_rate},"
                f" but found {sample_rate} instead.")
        if data.shape[1] != expected_num_channels:
            raise ValueError(f"From {filename}, expected {expected_num_channels} channels,"
                f" but found {data.shape[1]} instead.")

        # Complain if there's a length mismatch longer than about a second.
        if abs(data.shape[0] - expected_num_samples) > expected_sample_rate:
            raise ValueError(f"From {filename}, got {data.shape[0]}"
              f" samples instead of {expected_num_samples}.")

        # If there's a small mismatch, just patch it.
        data = patch_audio_length(data, expected_num_samples)

        return data

    # Not a format we can read directly.  Instead, use ffmpeg to get it
    # indirectly.  (...or simply pull it from the cache, if it happens to be
    # there.)
    cached_filename, success = cache.lookup(filename, 'flac')
    if not success:
        print(f'Extracting audio from {filename}')
        assert '.flac' in direct_formats
        full_filename = os.path.join(os.getcwd(), filename)
        with temporary_current_directory():
            audio_filename = 'audio.flac'
            ffmpeg( f'-i "{full_filename}"',
                    '-vn',
                    f'"{audio_filename}"')
            os.rename(audio_filename, cached_filename)
            cache.insert(cached_filename)

    return audio_samples_from_file(cached_filename,
                                   cache,
                                   expected_sample_rate,
                                   expected_num_channels,
                                   expected_num_samples)

def parse_subtitles(srt_text, subtitles_filename=None):
    """Parse a string of SRT subtitles into the form used in this library.

    :param srt_text: A string containing subtitle text in SRT format.
    :param subtitles_filename: An optional filename to include if an exception
            must be raised.
    :return: A generator that yields subtitles, each a `(start_time, end_time, text)`
            triple.

    """
    for subtitle in srt_text.split('\n\n'):
        lines = subtitle.split('\n')
        # Ignore lines[0], a sequence number which we don't need.
        times = lines[1]
        if match := re.match(r'(.*) --> (.*)', times):
            start = parse_hms_to_seconds(match.group(1))
            end = parse_hms_to_seconds(match.group(2))
        else:
            raise ValueError(f'Invalid time range {times} in {subtitles_filename}.')
        text = '\n'.join(lines[2:])
        yield (start, end, text)

def subtitles_from_file(filename, cache):
    """ Extract subtitles from a file.

    :param filename: The name of a media file that includes a subtitle stream.
    :param cache: A :class:`ClipCache` that might have the subtitle stream we
            want, or into which it can be stored.
    :return: A generator that yields subtitles, each a `(start_time, end_time, text)`
            triple.

    """

    # What file should the subtitles live in and does it exist already?
    subtitles_filename, exists = cache.lookup('subtitles',
                                              'srt',
                                              use_hash=False)

    # If we don't already have the subtitles file, use ffmpeg to get it.
    if not exists:
        print(f'Extracting subtitles from {filename}')
        with temporary_current_directory():
            ffmpeg( f'-i "{filename}"',
                   '-map 0:s:0',
                    f'"{subtitles_filename}"')

    # Read the subtitles in from the file.
    with open(subtitles_filename, 'r') as f:
        srt_text = f.read().strip()

    # Send 'em back.
    yield from parse_subtitles(srt_text)

def get_requested_intervals(requested_indices, max_gap):
    """For a given set of requested indices, return an iterable of start/stop
    pairs that covers everything that was requested.  If there are gaps smaller
    than max_gap, include those missing ones too."""
    if len(requested_indices)==0:
        return

    start = None

    a = sorted(requested_indices)
    b = itertools.pairwise(a)

    for i1, i2 in itertools.chain(b, [(a[-1], float('inf'))]):
        # Are we just getting started?
        if start is None:
            start = i1

        # Is the jump here too big to put into one interval?  If so, end the
        # previous interval and start a new one.
        if i2 - i1 > max_gap:
            yield (start, i1)
            start = i2



class from_file(Clip, FiniteIndexed):
    """A clip read from a file such as an mp4, flac, or other format readable
    by ffmpeg.

    :param filename: The source file to import.
    :param supress: A list containing some (possibly empty) subset of
            `"video"`, `"audio"`, and `"subtitle"`.  Streams of those types
            will be ignored.
    :param cache_dir: The directory to use for the frame cache.

    Details about what is included in the source file are extracted by parsing
    the output of `ffprobe`.  We make a best effort to sort out the metrics of
    the available video and audio streams, but encoded video is complicated, so
    there are surely many variations that are not yet handled correctly.  If
    you have a video for which this process fails, the maintainers would be
    interested to see it.

    Video is read by asking `ffmpeg` to "explode" the video stream into
    individual images.  This process takes some time, so the frames are cached
    for future runs.  There can potentially be lots of images, so you'll want
    to keep an eye on available disk space.

    |from-source|"""

    def __init__(self, filename, suppress=None, cache_dir=None):
        Clip.__init__(self)

        # Make sure the file exists.
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Could not open file {filename}.")
        self.filename = os.path.abspath(filename)

        self.file_timestamp = os.path.getmtime(self.filename)

        if cache_dir is None:
            cache_dir = os.path.join('/tmp/clipcache', re.sub('/', '_', self.filename))
        else:
            require_string(cache_dir, 'cache directory')

        self.cache = ClipCache(directory=cache_dir)

        self.acquire_metrics(suppress)

        if self.frame_rate:
            # Note: Might not have a frame rate if there's no video stream.
            FiniteIndexed.__init__(self,
                                   num_frames=self.metrics.length*self.frame_rate,
                                   length=self.metrics.length)

        self.samples = None
        self.subtitles = None

    def acquire_metrics(self, suppress=None):
        """ Set the metrics attribute, either by grabbing the metrics from the
        cache, or by getting them the hard way via ffprobe."""

        # Do we have the metrics in the cache?
        dimensions_filename, exists = self.cache.lookup(f'dimensions-{self.file_timestamp}',
                                                        'dim',
                                                        use_hash=False)

        if exists:
            # Yes. Grab it.
            print(f"Using cached dimensions for {self.filename}")
            with open(dimensions_filename, 'r') as f:
                deets = f.read()
        else:
            # No.  Get the metrics, then store in the cache for next time.
            print(f"Probing dimensions for {self.filename}")
            with subprocess.Popen(f'ffprobe -hide_banner -v error "{self.filename}" '
                                  '-of compact -show_entries stream '
                                  '-show_entries format ',
                                  shell=True,
                                  stdout=subprocess.PIPE) as proc:
                deets = proc.stdout.read().decode('utf-8')

            with open(dimensions_filename, 'w') as f:
                print(deets, file=f)
            self.cache.insert(dimensions_filename)

        # Parse the (very detailed) ffprobe response to get the metrics we
        # need.
        response = metrics_from_ffprobe_output(deets, self.filename, suppress)

        self.metrics, self.frame_rate, self.has_video, self.has_audio, self.has_subtitles = response

        self.requested_indices = set()

    def frame_signature(self, t):
        require_positive(t, "timestamp")

        if self.has_video:
            index = self.time_to_frame_index(t)
            return [self.filename, self.file_timestamp, index]
        else:
            return ['solid', {
                'width': self.metrics.width,
                'height': self.metrics.height,
                'color': [0,0,0,255]
            }]

    def request_frame(self, t):
        require_non_negative(t, "timestamp")
        if self.has_video:
            index = self.time_to_frame_index(t)
            self.requested_indices.add(index)

    def get_frame(self, t):
        require_non_negative(t, "timestamp")

        if not self.has_video:
            return np.zeros([self.metrics.height, self.metrics.width, 4], np.uint8)
        else:
            index = self.time_to_frame_index(t)
            filename, exists = self.cache.lookup(f'{index:06d}',
                                              self.cache.frame_format,
                                              use_hash=False)
            if not exists:
                self.explode()

            try:
                image = read_image(filename)
            except FileNotFoundError as fnfe:
                if index not in self.requested_indices:
                    raise ValueError(f'Tried to get frame at time {t} with index {index}, but '
                                      'the file does not exist, probably beacause the frame '
                                      'was not requested.') from fnfe
                else:
                    raise #pragma nocover

            return image


    def get_subtitles(self):
        if self.subtitles is None:
            if self.has_subtitles:
                self.subtitles = list(subtitles_from_file(self.filename, self.cache))
            else:
                self.subtitles = []
        return self.subtitles

    def explode_interval(self, start_index, end_index):
        """Expand the given range of frames into the cache.  Helper for explode()."""
        start_time = start_index / self.frame_rate
        length = (end_index - start_index + 1) / self.frame_rate
        num_frames_expected = end_index - start_index

        num_exploded = 0

        # Set up a callback to grab the extracted frames to the cache.
        # Add each frame that was extracted to the cache.
        def move_frames_to_cache(min_age=1):
            nonlocal num_exploded
            for filename in glob.glob('*.png'):
                age = time.time() - os.path.getmtime(filename)
                if age < min_age: continue
                file_index = int(re.search(r'\d*', filename).group(0)) - 1
                shifted_index = start_index + file_index
                new_filename, exists = self.cache.lookup(f'{shifted_index:06d}',
                                                      self.cache.frame_format,
                                                      use_hash=False)
                if not exists:
                    os.rename(filename, new_filename)
                    self.cache.insert(new_filename)
                    num_exploded += 1

        # Extract the frames into the current temporary directory.
        # Occasionally move those frames into the cache as we go.
        ffmpeg(f'-ss {start_time}',
               f'-t {length}',
               f'-i "{self.filename}"',
               f'-r {self.frame_rate}',
               '%06d.png',
               task=f'Exploding {os.path.basename(self.filename)}',
               num_frames=num_frames_expected,
               callback=move_frames_to_cache)

        # Now that the extraction is complete, grab anything left over.
        move_frames_to_cache(min_age=-1)

        # Done!
        return num_exploded


    def explode(self):
        """Expand the requested frames into our cache for later.

        Returns the number that were already in the cache, the number actually
        exploded, and the number of failed frames."""

        assert self.has_video

        num_cached = 0
        num_exploded = 0
        num_missing = 0

        if len(self.requested_indices) == 0:
            return 0

        # Figure out which requested indices are missing from the cache.
        needed_indices = set()
        for index in self.requested_indices:
            _, exists = self.cache.lookup(f'{index:06d}',
                                          self.cache.frame_format,
                                          use_hash=False)

            if not exists:
                needed_indices.add(index)
            else:
                num_cached += 1

        # Explode everything, in chunks.
        with temporary_current_directory():
            for start_index, end_index in get_requested_intervals(needed_indices, 100):
                num_exploded += self.explode_interval(start_index, end_index)

        # Make sure we got all of the frames we expected to get.
        for index in sorted(self.requested_indices):
            # If we get here, it means ffmpeg thought a frame should exist,
            # but that frame was ultimately not extracted.  This seems to
            # happen from mis-estimations of the video length, or sometimes
            # from simply missing frames.  To keep things rolling, let's
            # fill in a black frame instead.
            filename, exists = self.cache.lookup(f'{index:06d}',
                                              self.cache.frame_format,
                                              use_hash=False)
            if not exists: # pragma: no cover
                print(f"[Exploding {self.filename} did not produce frame index={index}. "
                      "Using black instead.]")
                fr = np.zeros([self.height(), self.width(), 3], np.uint8)
                cv2.imwrite(filename, fr)
                self.cache.insert(filename)
                num_missing += 1

        return num_cached, num_exploded, num_missing


    def get_samples(self):
        if self.samples is None:
            if self.has_audio:
                self.samples = audio_samples_from_file(self.filename,
                                                       self.cache,
                                                       self.sample_rate(),
                                                       self.num_channels(),
                                                       self.num_samples())
            else:
                self.samples = np.zeros([self.metrics.num_samples(), self.metrics.num_channels])
        return self.samples

