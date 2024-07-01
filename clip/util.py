""" Assorted utilities that defy finer classification. """

import contextlib
import hashlib
import os
import re
import tempfile

from .validate import is_iterable

def flatten_args(args):
    """ Given a list of arguments, flatten one layer of lists and other
    iterables. """
    ret = []
    for x in args:
        if is_iterable(x):
            ret += x
        else:
            ret.append(x)
    return ret

@contextlib.contextmanager
def temporarily_changed_directory(directory):
    """Create a context in which the current directory has been changed to the
    given one, which should exist already.  When the context ends, change the
    current directory back."""
    previous_current_directory = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous_current_directory)


@contextlib.contextmanager
def temporary_current_directory():
    """Create a context in which the current directory is a new temporary
    directory.  When the context ends, the current directory is restored and
    the temporary directory is vaporized."""
    with tempfile.TemporaryDirectory() as td:
        with temporarily_changed_directory(td):
            try:
                yield
            finally:
                pass

def sha256sum_file(filename):
    """ Return a short hexadecmial hash of the contents of a file. """
    # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()

def format_seconds_as_hms(seconds):
    """Return a string representing the given time in 00:01:23,456 format.
    This specific format is important for saving subtitles in the format that
    ffmpeg likes to see."""
    seconds, fraction = divmod(seconds, 1)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    millis = int(1000*fraction)
    seconds = int(seconds)
    minutes = int(minutes)
    hours = int(hours)
    return f'{hours:02}:{minutes:02}:{seconds:02},{millis:03}'

def parse_hms_to_seconds(hms):
    """Parse a string in 00:01:23,456 into a floating point number of
    seconds."""
    if match := re.match(r"(\d\d):(\d\d):(\d\d),(\d\d\d)", hms):
        hours = float(match.group(1))
        mins = float(match.group(2))
        secs = float(match.group(3))
        millis = float(match.group(4))
        return millis/1000 + secs + 60*mins + 60*60*hours
    else:
        raise ValueError(f'Cannot parse {hms} as hours, minutes, seconds, and milliseconds.')
