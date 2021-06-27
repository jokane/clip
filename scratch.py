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

x = solid([0,255,0], 640, 480, 30, 5)
x = join(x, sine_wave(440, 0.25, x.length(), 48000, 2))
y = solid([0,0,255], 640, 480, 30, 5)
y = join(y, sine_wave(880, 0.25, y.length(), 48000, 2))
z = chain(x, y, fade=2)
z.save("twotone.mp4")

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


# x = from_file("music.mp3")
# x = scale_volume(x, 0.2)
# x.save("softmusic.mp4")

# x = from_file("intro.mp4")
# x = crop(x, [10, 10], [100, 100])
# x = slice_clip(x, 5, 10)
# x.save("cropped.mp4")

# font = "ethnocentric_rg_it.ttf"
# x = draw_text("Hello!", font, font_size=200, frame_rate=30, length=5)
# x.verify()
# x.save("hi.mp4")

x = solid([0,255,0], 640, 480, 30, 5)
x = join(x, sine_wave(440, 0.25, x.length(), 48000, 2))
y = solid([0,0,255], 640, 480, 30, 5)
y = join(y, sine_wave(880, 0.25, y.length(), 48000, 2))
z = chain(x, y, fade=2)
z.save("twotone.mp4")

# x = from_file("books.mp4", decode_chunk_length=None)
# x = to_monochrome(x)
# pprint(x.frame_signature(45))
# x.preview()

# x = static_image("samples/flowers.png", 30, 10)
# x = scale_by_factor(x, 0.2)
# 
# y = static_image("samples/flowers.png", 30, 10)
# y = scale_by_factor(y, 0.2)
# 
# z = composite(
#   Element(x, 1, [10, 10], Element.VideoMode.BLEND),
#   Element(y, 2, [100, 100], Element.VideoMode.BLEND),
#   Element(x, 3, [0, 100], Element.VideoMode.BLEND),
# )
# 
# z.save("blended.mp4")

# x = from_file("samples/books.mp4", decode_chunk_length=7)
# x = slice_clip(x, 0, 7)
# x = fade_out(x, 2)
# x.verify()
# # x.stage('frames')
# x.save("fade_out.mp4")
