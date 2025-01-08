# This is an attempt to find a reasonably good value for the max_gap when
# exploding a file.
#
# The question is: When there is a chunk of frames that we don't need, is it
# better to have ffmepg explode them anyway and ignore them, or to start a
# separate one?
#
# Idea: Model the time it takes to ffmpeg-explode a chunk of a video as linear
# function, including a fixed (i.e. not depending on the number of frames)
# amount of time A for starting and stopping along with a fixed amount of time
# B for each frame.  So to explode n frames, it would should take about A+Bn
# seconds of ffmpeg run time.
#
# Or, in terms of video time: To get t seconds of a video that runs at r frames
# per second, we should expect to wait A + Brt seconds.
#
# Q: When should we split into two ffmpeg calls for a gap in the requested
# frames?
# A: If n is the gap length, we should stop when Bn > A, i.e. when the marginal
# time to extract those frames exceeds the time to stop and start a new ffmpeg.
# So when n > A/B.
#
#
# Anyhoo... This script is an attempt to estimate A and B.
# 
# Of course, the real values of A and B will vary based on loads of factors,
# including probably things like the resolution and codec of the video, the
# computer we're running on, and who knows what else.  But perhaps this can get
# us in the right order of magnitude.
#

from clip import *
import time
import random
import os
import numpy
import matplotlib.pyplot as plt

# Prolly need to do a 'make test' first to download this.
TEST_FILES_DIR = os.path.join(os.path.split(__file__)[0], "..", "test", ".test_files")

results = []
for _ in range(100):
    t = random.uniform(5, 120)
    with temporary_current_directory():
        start = time.perf_counter()
        x = from_file(f'{TEST_FILES_DIR}/bunny.webm', cache_dir=os.getcwd())
        y = slice_clip(x, 0, t)
        y.request_all_frames(100)
        x.explode()
        end = time.perf_counter()
        results.append((t, end-start))

print(sorted(results))
xs, ys = zip(*results)
plt.scatter(xs, ys)
plt.show()
