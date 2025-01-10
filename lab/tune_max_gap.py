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
# Q: When should we split into two ffmpeg calls for a gap in the requested
# frames?
# A: If n is the gap length, we should stop when Bn > A, i.e. when the marginal
# time to extract those frames exceeds the time to stop and start a new ffmpeg.
# So when n > A/B.
#
#
# Anyhoo... This script is an attempt to estimate A and B.
#
# It gave A=0.16 and B=0.0019, with A/B=84.6.
# 
# Of course, the real values of A and B will vary based on loads of factors,
# including probably things like the resolution and codec of the video, the
# computer we're running on, and who knows what else.  But perhaps this can get
# us in the right order of magnitude.
#

from clip import *
import time
import pickle
import random
import os
import numpy
import matplotlib.pyplot as plt
import pprint

# Prolly need to do a 'make test' first to download bunny.webm to this folder.
TEST_FILES_DIR = os.path.join(os.path.split(__file__)[0], "..", "test", ".test_files")

if not os.path.exists('data.pkl'):
    results = []
    for t in numpy.linspace(start=5, stop=120, num=200):
        with temporary_current_directory():
            start = time.perf_counter()
            x = from_file(f'{TEST_FILES_DIR}/bunny.webm', cache_dir=os.getcwd())
            y = slice_clip(x, 0, t)
            y.request_all_frames(x.frame_rate)
            x.explode()
            end = time.perf_counter()
            results.append((t*x.frame_rate, end-start))

    with open('data.pkl', 'wb') as f:
        pickle.dump(results, f)
else:
    with open('data.pkl', 'rb') as f:
        results = pickle.load(f)

# For each trial, results has: (number of frames, real time to explode)
xs, ys = zip(*results)

fit = np.polynomial.polynomial.Polynomial.fit(xs, ys, 1).convert()

A = fit.coef[0]
B = fit.coef[1]

print(f'Exploding n frames of video takes approximately {A:.02}+{B:.02}n seconds of real time.')
print(f'It is worth starting a new explode() for gaps that are at least {A/B} frames.')

plt.scatter(xs, ys)
plt.show()
