""" Assorted utilities that defy finer classification. """

import contextlib
import hashlib
import os
import re
import tempfile

import cv2
import numpy as np

from .validate import is_iterable

def flatten_args(stuff):
    """ Given a list of arguments, flatten one layer of lists and other
    iterables.

    :param stuff: A list of any sort of stuff.
    :return: A list containing a flattened version of the `stuff`.  For
            anything iterable in the `stuff`, replace that iterable with the
            items it yields.

    """
    ret = []
    for x in stuff:
        if is_iterable(x):
            ret += x
        else:
            ret.append(x)
    return ret

@contextlib.contextmanager
def temporarily_changed_directory(directory):
    """A context in which the current directory has been changed to the
    given one, which should exist already.

    When the context ends, change the current directory back."""
    previous_current_directory = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous_current_directory)


@contextlib.contextmanager
def temporary_current_directory():
    """A context in which the current directory is a new temporary
    directory.

    When the context begins, a new temporary directory is created.  This new
    directory becomes the current directory.

    When the context ends, the current directory is restored and the temporary
    directory is vaporized."""
    with tempfile.TemporaryDirectory() as td:
        with temporarily_changed_directory(td):
            try:
                yield
            finally:
                pass

def sha256sum_file(filename):
    """ Hash the contents of a file.

    :param filename: The name of a file to read.
    :return: A short hexadecmial hash of the contents of a file.

    This implementation uses `sha256`.

    """
    # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    h = hashlib.sha256()
    b = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            h.update(mv[:n])
    return h.hexdigest()

def save_subtitles(subtitles, destination):
    """Save a list of subtitles to an SRT file.

    :param subtitles: A list of `(start_time, end_time, text)` triples.
    :param destination: A string filename or file-like object.

    """
    with contextlib.ExitStack() as exst:
        if isinstance(destination, str):
            f = exst.enter_context(open(destination, 'w'))
        else:
            f = destination

        for number, subtitle in enumerate(subtitles):
            print(number+1, file=f)
            print(format_seconds_as_hms(subtitle[0]),
                  '-->',
                  format_seconds_as_hms(subtitle[1]),
                  file=f)
            print(subtitle[2], file=f)
            print(file=f)

def format_seconds_as_hms(seconds):
    """Format a float number of seconds in the format that `ffmpeg` likes to
    see for subtitles.

    :param seconds: A float number of seconds.
    :return: The given time in `00:01:23,456` format.

    """
    seconds, fraction = divmod(seconds, 1)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    millis = int(1000*fraction)
    seconds = int(seconds)
    minutes = int(minutes)
    hours = int(hours)
    return f'{hours:02}:{minutes:02}:{seconds:02},{millis:03}'

def parse_hms_to_seconds(hms):
    """Parse a string in the format that `ffmpeg` uses for subtitles into a
    float number of seconds.

    :param hms: The given time in `00:01:23,456` format.
    :return: A float number of seconds for the input string.

    """

    if match := re.match(r"(\d\d):(\d\d):(\d\d),(\d\d\d)", hms):
        hours = float(match.group(1))
        mins = float(match.group(2))
        secs = float(match.group(3))
        millis = float(match.group(4))
        return millis/1000 + secs + 60*mins + 60*60*hours
    else:
        try:
            return float(hms)
        except:
            raise ValueError(f'Cannot parse {hms} as hours, minutes, seconds, and milliseconds.')

def read_image(filename):
    """Read an image from disk.  If needed, convert it to the correct RGBA
    uint8 format.

    :param filename: The name of the file to read.
    :return: The image data from that file, in RGBA uint8 format.

    """

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Trying to open {filename}, which does not exist. "
                                f"(Current working directory is {os.getcwd()}")
    frame = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    assert frame is not None
    assert len(frame.shape) == 3
    if frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    assert frame.shape[2] == 4, frame.shape
    assert frame.dtype == np.uint8
    return frame



def func_signature(func):
    """Return a short signature string for a function, aiming to be different
    when the function's behavior changes.  Accounts for code changes in the
    function, in those it calls, and in closure of this function.

    It's not hard to break this, of course.  (Examples: mutable closure values,
    I/O, etc, etc.)  But it should cover many common cases.

    :param function: A function for which to generate a signature.

    """
    addr_re = re.compile(r'0x[0-9a-fA-F]+')
    def hide_pointers(x):
        return addr_re.sub('0xXX', repr(x))

    parts = []
    stack = [func.__code__]
    while stack:
        c = stack.pop()
        parts.append(c.co_code)
        parts.append(repr(c.co_names).encode())
        parts.append(repr(c.co_varnames).encode())
        parts.append(repr(c.co_freevars).encode())
        for const in c.co_consts:
            if hasattr(const, 'co_code'):
                stack.append(const)
            else:
                parts.append(hide_pointers(const).encode())

    if func.__closure__:
        for cell in func.__closure__:
            try:
                r = repr(cell.cell_contents)
                parts.append(hide_pointers(r).encode())
            except ValueError:
                parts.append(b'<empty cell>')

    parts.append(hide_pointers(func.__defaults__).encode())
    parts.append(hide_pointers(func.__kwdefaults__).encode())

    return hashlib.sha1(b''.join(parts)).hexdigest()[:7]
