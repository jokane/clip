""" Tools for slicing and cropping clips, across time and space respectively. """

from .base import Clip, MutatorClip, require_clip
from .metrics import Metrics
from .chain import chain
from .validate import (require_int, require_non_negative, require_less_equal, require_less,
                       require_float, require_positive)

class crop(MutatorClip):
    """Trim the frames of a clip to show only the rectangle between
    lower_left and upper_right."""
    def __init__(self, clip, lower_left, upper_right):
        super().__init__(clip)
        require_int(lower_left[0], "lower left")
        require_non_negative(lower_left[0], "lower left")
        require_int(lower_left[1], "lower left")
        require_non_negative(lower_left[1], "lower left")
        require_int(upper_right[0], "upper right")
        require_less_equal(upper_right[0], clip.width(), "upper right", "width")
        require_int(upper_right[1], "upper right")
        require_less_equal(upper_right[1], clip.height(), "upper right", "width")
        require_less(lower_left[0], upper_right[0], "lower left", "upper right")
        require_less(lower_left[1], upper_right[1], "lower left", "upper right")

        self.lower_left = lower_left
        self.upper_right = upper_right

        self.metrics = Metrics(self.metrics,
                               width=upper_right[0]-lower_left[0],
                               height=upper_right[1]-lower_left[1])

    def frame_signature(self, t):
        return ['crop', self.lower_left, self.upper_right, self.clip.frame_signature(t)]

    def get_frame(self, t):
        frame = self.clip.get_frame(t)
        ll = self.lower_left
        ur = self.upper_right
        return frame[ll[1]:ur[1], ll[0]:ur[0], :]

class slice_clip(MutatorClip):
    """ Extract the portion of a clip between the given times. Endpoints
    default to the start and end of the clip."""
    def __init__(self, clip, start=0, end=None):
        super().__init__(clip)
        if end is None:
            end = self.clip.length()

        require_float(start, "start time")
        require_non_negative(start, "start time")
        require_float(end, "end time")
        require_non_negative(end, "end time")
        require_less_equal(start, end, "start time", "end time")
        require_less_equal(end, clip.length(), "start time", "end time")

        self.start_time = start
        self.end_time = end
        self.start_sample = int(start * self.sample_rate())
        self.metrics = Metrics(self.metrics, length=end-start)

        self.subtitles = None

    def frame_signature(self, t):
        return self.clip.frame_signature(self.start_time + t)

    def request_frame(self, t):
        self.clip.request_frame(self.start_time + t)

    def get_frame(self, t):
        return self.clip.get_frame(self.start_time + t)

    def get_samples(self):
        original_samples = self.clip.get_samples()
        return original_samples[self.start_sample:self.start_sample+self.num_samples()]

    def get_subtitles(self):
        if self.subtitles is None:
            self.subtitles = []
            for subtitle in self.clip.get_subtitles():
                new_start = subtitle[0] - self.start_time
                new_end = subtitle[1] - self.start_time
                length = self.length()
                if 0 <= new_start <= length or 0 <= new_end <= length:
                    new_start = max(0, new_start)
                    new_end = min(self.length(), new_end)
                    self.subtitles.append((new_start, new_end, subtitle[2]))
        return self.subtitles

def slice_out(clip, start, end) -> Clip:
    """ Remove the part between the given endponts. """
    require_clip(clip, "clip")
    require_float(start, "start time")
    require_non_negative(start, "start time")
    require_float(end, "end time")
    require_positive(end, "end time")
    require_less(start, end, "start time", "end time")
    require_less_equal(end, clip.length(), "end time", "clip length")

    # Special cases because slice_clip complains if we ask for a length zero
    # slice.  As a bonus, this eliminates the need for chaining. (And, if for
    # some reason, we're asked to slice out the entire clip, the aforementioned
    # slice_clip complaint will kick in.)
    if start == 0:
        return slice_clip(clip, end, clip.length())
    elif end == clip.length():
        return slice_clip(clip, 0, start)

    # Normal case: Something before and something after.
    return chain(slice_clip(clip, 0, start),
                 slice_clip(clip, end, clip.length()))
