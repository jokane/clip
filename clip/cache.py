""" An object representing an on-disk cache of various pre-computed stuff. """

import collections
import hashlib
import os
import shutil


class ClipCache:
    """An object for managing the cache of files.  This might contain
    already-computed frames, audio segments, and other things.

    :param directory: The cache directory.
    :param frame_format: The file extension to use for cached image frames.
            This is not used directly in this class, but maybe referenced by
            other code that uses the cache.
    """

    def __init__(self, directory, frame_format='png'):
        self.directory = directory
        self.cache = None
        self.frame_format = frame_format

        assert self.directory[0] == '/', \
          'Creating cache with a relative path.  This can cause problems later if the ' \
          'current directory changes, which is not unlikely to happen.'

    def scan_directory(self):
        """Examine the cache directory and remember what we see there."""
        self.cache = {}
        try:
            for cached_file in os.listdir(self.directory):
                self.cache[os.path.join(self.directory, cached_file)] = True
        except FileNotFoundError:
            os.makedirs(self.directory)

        counts = '; '.join(map(lambda x: f'{x[1]} {x[0]}',
          collections.Counter(map(lambda x: os.path.splitext(x)[1][1:],
          self.cache.keys())).items()))
        print(f'Found {len(self.cache)} cached items ({counts}) in {self.directory}')

    def clear(self):
        """ Delete all the files in the cache. """
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)
        self.cache = None

    def sig_to_fname(self, sig, ext, use_hash=True):
        """Compute the filename where something with the given signature and
        extension should live.

        :param sig: A unique signature for the thing to be stored.  This will
                be converted to a string and probably hashed to build the
                filename. Usually it will be the output of the
                :func:`~clip.Clip.frame_signature` of some :class:`Clip` class.

        :param ext: A string extension for the file.

        :param use_hash: Should we hash the `sig`, or just stringify it?  Use
                `False` to get files with more readable names.

        :return: A full pathname within the the cache directory telling whether
                an object with the given signature should live.
        """
        if use_hash:
            blob = hashlib.md5(str(sig).encode()).hexdigest()
        else:
            blob = str(sig)
        return os.path.join(self.directory, f'{blob}.{ext}')

    def lookup(self, sig, ext, use_hash=True):
        """Determine the appropriate filename for something with the given
        signature and extension.  Also determine whether that file exists already.

        :param sig: A unique signature for the thing to be stored.  This will
                be converted to a string and possibly hashed to build the
                filename. Usually it will be the output of the
                :func:`~clip.Clip.frame_signature` of some :class:`Clip` class.

        :param ext: A string extension for the file.

        :param use_hash: Should we hash the `sig`, or just stringify it?  Use
                `False` to get files with more readable names.

        :return: A tuple of the filename, along with `True` or `False`
                indicating whether that file exists or not.

        """
        if self.cache is None: self.scan_directory()
        cached_filename = self.sig_to_fname(sig, ext, use_hash)
        return (cached_filename, cached_filename in self.cache)

    def insert(self, filename):
        """Update the cache to reflect the fact that the given file exists.

        :param filename: The name of the file to insert.

        """
        if self.cache is None: self.scan_directory()
        self.cache[filename] = True

