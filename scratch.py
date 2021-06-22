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
# cache.clear()
# 
# x = solid([0,255,0], 640, 480, 30, 5)
# x = join(x, sine_wave(440, 0.25, x.length(), 48000, 2))
# 
# y = solid([0,0,255], 640, 480, 30, 5)
# y = join(y, sine_wave(880, 0.25, y.length(), 48000, 2))
# 
# #z = chain(x, y)
# 
# z = fade_chain(2, x, y)
# 
# z.save("twotone.mp4")

# x = from_file("books.mp4", decode_chunk_length=None)
# x.save("new_books.mp4")

# x = from_file("new_books.mp4", decode_chunk_length=None)
# x.verify()
# x.save("newer_books.mp4")

# x = from_file("music.mp3")
# x.verify()
# x.save("music.mp3")

# x = from_file("intro.mp4")
# x = slice_clip(x, start=80)
# x.save("sliced.mp4")

# x = from_file("intro.mp4", decode_chunk_length=None)
# x = reverse(x)
# x.save("reversed.mp4")


x = from_file("music.mp3")
x = scale_volume(x, 0.2)
x.save("softmusic.mp4")
