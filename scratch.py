#!/usr/bin/env python3

from clip3 import *

# solid([0,0,0], 640, 480, 30, 300]).save('black.mp4')
# solid([255,0,0], 640, 480, 30, 300).save('red.mp4')
# solid([0,255,0], 640, 480, 30, 300).save('green.mp4')
# solid([0,0,255], 640, 480, 30, 300).save('blue.mp4')
# solid([255,255,255], 640, 480, 30, 300).save('white.mp4')


# temporal_composite(
#   (solid(640, 480, 30, 5, [0,0,0]), 0),
#   (solid(640, 480, 30, 5, [255,0,0]), 4.5)
# ).save('tc.mp4')

x = sine_wave(440, 0.25, 5, 48000, 2)
x.save("A4.flac")

y = solid([0,255,0], 640, 480, 30, 5)

join(y, x).save("green-A4.mp4")

