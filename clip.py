"""
clip.py
--------------------------------------------------------------------------------
This is a library for manipulating and generating short video clips.  It can
read video in any format supported by ffmpeg, and provides abstractions for
things like cropping, superimposing, adding text, trimming, fading in and out,
fading between clips, etc.  Additional effects can be achieved by filtering the
frames through custom functions.

The basic currency is the (abstract) Clip class, which encapsulates a stream of
identically-sized frames, each a cv2 image.  Clips may be created by reading
from a video file (see: video_file), by starting from a blank video (see:
solid), or by using one of the other subclasses or functions to modify existing
clips.  All of these methods are non-destructive: Doing, say, a crop() on a
clip with return a new, cropped clip, but will not affect the original.  Each
clip also has an audio track, which must the same length as the video part.

Possibly relevant implementation details:
- There is a "lazy evaluation" flavor to the execution.  Simply creating a
  clip object will check for some errors (for example, mismatched sizes or
  framerates) but will not actually do any work to produce the video.  The real
  rendering happens when one calls one of the save or play methods on a clip.

- To accelerate things across multiple runs, frames are cached in
  /tmp/clipcache.  This caching is done based on a string "signature" for each
  frame, which is meant to uniquely identify the visual contents of a frame.

- Some things are subclasses of Clip, whereas other parts are just functions.
  However, all of them use snake_case because it should not matter to a user
  whether it's a subclass or a function.

--------------------------------------------------------------------------------

"""

# pylint: disable=too-many-lines

from abc import ABC
import contextlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time

import progressbar

def is_float(x):
    """ Can the given value be interpreted as a float? """
    try:
        float(x)
        return True
    except TypeError:
        return False
    except ValueError:
        return False

def is_int(x):
    """ Can the given value be interpreted as an int? """
    return isinstance(x, int)

def is_string(x):
    """ Is the given value actually a string? """
    return isinstance(x, str)

def is_positive(x):
    """ Can the given value be interpreted as a positive number? """
    return x>0

def is_even(x):
    """ Is it an even number? """
    return x%2 == 0

def is_non_negative(x):
    """ Can the given value be interpreted as a non-negative number? """
    return x>=0

def is_color(color):
    """ Is this a color, in RGB 8-bit format? """
    try:
        if len(color) != 3: return False
    except TypeError:
        return False
    if not is_int(color[0]): return False
    if not is_int(color[1]): return False
    if not is_int(color[2]): return False
    if color[0] < 0 or color[0] > 255: return False
    if color[1] < 0 or color[1] > 255: return False
    if color[2] < 0 or color[2] > 255: return False
    return True

def is_int_point(pt):
    """ Is this a 2d point with integer coordinates. """
    if len(pt) != 2: return False
    if not is_int(pt[0]): return False
    if not is_int(pt[1]): return False
    return True

def is_iterable(x):
    """ Is this a thing that can be iterated? """
    try:
        iter(x)
        return True
    except TypeError:
        return False

def require(x, func, condition, name, exception_class):
    """ Make sure func(x) returns a true value, and complain if not."""
    if not func(x):
        raise exception_class(f'Expected {name} to be a {condition}, '
                              f'but got a {type(x)} with value {x} instead.')

def require_int(x, name):
    """ Raise an informative exception if x is not an integer. """
    require(x, is_int, "integer", name, TypeError)

def require_float(x, name):
    """ Raise an informative exception if x is not a float. """
    require(x, is_float, "float", name, TypeError)

def require_string(x, name):
    """ Raise an informative exception if x is not a string. """
    require(x, is_string, "string", name, TypeError)

def require_clip(x, name):
    """ Raise an informative exception if x is not a Clip. """
    require(x, lambda x: isinstance(x, Clip), "Clip", name, TypeError)

def require_color(x, name):
    """ Raise an informative exception if x is not a color. """
    require(x, is_color, "color", name, TypeError)

def require_int_point(x, name):
    """ Raise an informative exception if x is not a integer point. """
    require(x, is_int_point, "point with integer coordinates", name, TypeError)

def require_positive(x, name):
    """ Raise an informative exception if x is not positive. """
    require(x, is_positive, "positive number", name, ValueError)

def require_even(x, name):
    """ Raise an informative exception if x is not even. """
    require(x, is_even, "even", name, ValueError)

def require_non_negative(x, name):
    """ Raise an informative exception if x is not 0 or positive. """
    require(x, is_non_negative, "non-negative", name, ValueError)

def require_equal(x, y, name):
    """ Raise an informative exception if x and y are not equal. """
    if x != y:
        raise ValueError(f'Expected {name} to be equal, but they are not.  {x} != {y}')

def require_less_equal(x, y, name1, name2):
    """ Raise an informative exception if x is not less than or equal to y. """
    if x > y:
        raise ValueError(f'Expected "{name1}" to be less than or equal to "{name2}",'
          f' but it is not. {x} > {y}')

def require_less(x, y, name1, name2):
    """ Raise an informative exception if x is greater than y. """
    if x >= y:
        raise ValueError(f'Expected "{name1}" to be less than "{name2}", '
          f'but it is not. {x} >= {y}')

def require_callable(x, name):
    """ Raise an informative exception if x is not callable. """
    require(x, callable, "callable", name, TypeError)

class FFMPEGException(Exception):
    """Raised when ffmpeg fails for some reason."""

def ffmpeg(*args, task=None, num_frames=None):
    """Run ffmpeg with the given arguments.  Optionally, maintain a progress
    bar as it goes."""

    with tempfile.NamedTemporaryFile() as stats:
        command = f"ffmpeg -y -vstats_file {stats.name} {' '.join(args)} 2> errors"
        with subprocess.Popen(command, shell=True) as proc:
            t = threading.Thread(target=proc.communicate)
            t.start()

            if task is not None:
                with custom_progressbar(task=task, steps=num_frames) as pb:
                    pb.update(0)
                    while proc.poll() is None:
                        try:
                            with open(stats.name) as f: #pragma: no cover
                                fr = int(re.findall(r'frame=\s*(\d+)\s', f.read())[-1])
                                pb.update(min(fr, num_frames-1))
                        except FileNotFoundError:
                            pass # pragma: no cover
                        except IndexError:
                            pass # pragma: no cover
                        time.sleep(1)

            t.join()

            if os.path.exists('errors'):
                shutil.copy('errors', '/tmp/ffmpeg_errors')
                with open('/tmp/ffmpeg_command', 'w') as f:
                    print(command, file=f)

            if proc.returncode != 0:
                if os.path.exists('errors'):
                    with open('errors', 'r') as f:
                        errors = f.read()
                else:
                    errors = '[no errors file found]' #pragma: no cover
                message = (
                  f'Alas, ffmpeg failed with return code {proc.returncode}.\n'
                  f'Command was: {command}\n'
                  f'Standard error was:\n{errors}'
                )
                raise FFMPEGException(message)

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


def custom_progressbar(task, steps):
    """Return a progress bar (for use as a context manager) customized for
    our purposes."""
    digits = int(math.log10(steps))+1
    widgets = [
        '|',
        f'{task:^25s}',
        ' ',
        progressbar.Bar(),
        progressbar.Percentage(),
        '| ',
        progressbar.SimpleProgress(format=f'%(value_s){digits}s/%(max_value_s){digits}s'),
        ' |',
        progressbar.ETA(
            format_not_started='',
            format_finished='%(elapsed)8s',
            format='%(eta)8s',
            format_zero='',
            format_NA=''
        ),
        '|'
    ]
    return progressbar.ProgressBar(max_value=steps, widgets=widgets)


class Clip(ABC):
    """The base class for all clips.  A finite series of frames, each with
    identical height and width, meant to be played at a given rate, along with
    an audio clip of the same length."""

