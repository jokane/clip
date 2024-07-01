""" An object representing an on-disk cache of various pre-computed stuff. """

import collections
import hashlib
import os
import shutil


class ClipCache:
    """An object for managing the cache of already-computed frames, audio
    segments, and other things."""
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
        extension should live."""
        if use_hash:
            blob = hashlib.md5(str(sig).encode()).hexdigest()
        else:
            blob = str(sig)
        return os.path.join(self.directory, f'{blob}.{ext}')

    def lookup(self, sig, ext, use_hash=True):
        """Determine the appropriate filename for something with the given
        signature and extension.  Return a tuple with that filename followed
        by True or False, indicating whether that file exists or not."""
        if self.cache is None: self.scan_directory()
        cached_filename = self.sig_to_fname(sig, ext, use_hash)
        return (cached_filename, cached_filename in self.cache)

    def insert(self, fname):
        """Update the cache to reflect the fact that the given file exists."""
        if self.cache is None: self.scan_directory()
        self.cache[fname] = True

