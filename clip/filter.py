""" A tool for filtering the frames of a clip. """

import dis
import hashlib
import inspect

from .base import MutatorClip
from .metrics import Metrics
from .validate import require_callable, require_int, require_positive

class filter_frames(MutatorClip):
    """A clip formed by passing the frames of another clip through some
    function. |modify

    :param clip: The clip to filter.
    :param func: The filter function.
    :param name: A name for the filter.
    :param size: The size of the output frames, on None

    For `func`, provide a callable that takes either one or two arguments.

        - If `func` takes one argument, the argument will be the frame itself.

        - If `func` takes two arguments, the arguments frame and its time index.

    In either case, `func` should return the output frame.

    The `name` is an optional string.  If a name is given, it is included in
    the frame signatures.  This can help with debugging.

    Output frames may have a different size from the input ones, but must all
    be the same size across the whole clip.  The `size` parameter specifies the
    size of the output frames.  For `size`, use either `None`, a
    `(width,height)` tuple, or the string `"same"`.

        - Set `size` to `None` to infer the width and height of the result by
          executing the filter function on a sample frame.  This can be slow
          if, for example, `clip` is (or relies upon) `from_file` clip or other
          time-intensive source.

        - Set size to a tuple of two positive integers `(width, height)` if you
          know them.  This avoids generating a sample frame.

        - Set size to `"same"` to assume the size is the same as the source
          clip. This avoids generating a sample frame.

    Audio remains unchanged from the original `clip`.
    
    """

    def __init__(self, clip, func, name=None, size=None):
        super().__init__(clip)

        require_callable(func, "filter function")
        self.func = func

        # Use the details of the function's bytecode to generate a "signature",
        # which we'll use in the frame signatures.  This should help to prevent
        # the need to clear cache if the implementation of a filter function is
        # changed.
        bytecode = dis.Bytecode(func, first_line=0)
        description = bytecode.dis()
        self.sig = hashlib.sha1(description.encode('UTF-8')).hexdigest()[:7]


        # Acquire a name for the filter.
        if name is None:
            name = self.func.__name__
        self.name = name

        # Figure out if the function expects the index or not.  If not, wrap it
        # in a lambda to ignore the index.  But remember that we've done this,
        # so we can leave the index out of our frame signatures.
        parameters = list(inspect.signature(self.func).parameters)
        if len(parameters) == 1:
            self.depends_on_time = False
            def new_func(frame, t, func=self.func): #pylint: disable=unused-argument
                return func(frame)
            self.func = new_func
        elif len(parameters) == 2:
            self.depends_on_time = True
        else:
            raise TypeError(f"Filter function should accept either (frame) or "
                            f"(frame, t), not {parameters}.)")

        # Figure out the size.
        if size is None:
            sample_frame = self.func(clip.get_frame(0), 0)
            height, width, _ = sample_frame.shape
        else:
            try:
                width, height = size
                require_int(width, 'width')
                require_positive(width, 'width')
                require_int(height, 'height')
                require_positive(height, 'height')
            except ValueError as e:
                if size == "same":
                    width = clip.width()
                    height = clip.height()
                else:
                    raise ValueError(f'In filter_frames, did not understand size={size}.') from e

        self.metrics = Metrics(src = clip.metrics,
                               width = width,
                               height = height)

    def frame_signature(self, t):
        assert t < self.length()
        return [f"{self.name} (filter:{self.sig})", {
          't' : t if self.depends_on_time else None,
          'width' : self.width(),
          'height' : self.height(),
          'frame' : self.clip.frame_signature(t)
        }]

    def get_frame(self, t):
        return self.func(self.clip.get_frame(t), t)

