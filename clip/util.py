""" Assorted utilities that defy finer classification. """

import contextlib
import hashlib
import os
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

