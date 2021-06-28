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
# y = solid([0,0,255], 640, 480, 30, 5)
# y = join(y, sine_wave(880, 0.25, y.length(), 48000, 2))
# z = chain(x, y, fade=2)
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

# x = solid([0,255,0], 640, 480, 30, 5)
# x = join(x, sine_wave(440, 0.25, x.length(), 48000, 2))
# y = solid([0,0,255], 640, 480, 30, 5)
# y = join(y, sine_wave(880, 0.25, y.length(), 48000, 2))
# z = chain(x, y, fade=2)
# z.save("twotone.mp4")

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

# x = from_file("/tmp/VID_20200818_161859.mp4")
# print(x.metrics)
# x.verify(verbose=True)

# x = static_image("samples/flowers.png", 30, 5)
# x = scale_by_factor(x, 0.4)
# 
# y = spin(x, 2)
# 
# z = composite(
#   Element(x, 0, [50, 50], video_mode=Element.VideoMode.BLEND),
#   Element(y, 0, [250, 150], video_mode=Element.VideoMode.BLEND),
#   width=640,
#   height=480
# )
# z.preview()

# font = "samples/ethnocentric_rg_it.ttf"
# x = vstack(
#   draw_text("Hello,", font, font_size=200, frame_rate=30, length=5),
#   vstack(
#     draw_text("hello", font, font_size=20, frame_rate=30, length=5),
#     align=Align.CENTER,
#     width=2000
#   ),
#   draw_text("World!", font, font_size=200, frame_rate=30, length=5),
#   align=Align.RIGHT,
#   width=2000
# )
# 
# x.preview()
# x.save("hi.mp4")

# font = "samples/ethnocentric_rg_it.ttf"
# x = draw_text("Hello,", font, font_size=200, frame_rate=30, length=5)
# y = draw_text("World,", font, font_size=200, frame_rate=30, length=5)
# x = composite(
#   Element(x, 0, [0,0], Element.VideoMode.REPLACE),
#   Element(y, 0, [25,25], Element.VideoMode.BLEND)
# )
# x.preview()

# font = "samples/ethnocentric_rg.ttf"
# x = draw_text("red", font, font_size=200, color=(255,0,0), frame_rate=30, length=5)
# y = draw_text("green", font, font_size=200, color=(0,255,0), frame_rate=30, length=5)
# z = draw_text("blue", font, font_size=200, color=(0,0,255), frame_rate=30, length=5)
# x = vstack(x, y, z)
# x.preview()

x = from_file("samples/books.mp4")
y = resample(from_file("samples/bunny.webm"), frame_rate=x.frame_rate(), sample_rate=x.sample_rate())

z = superimpose_center(x, y, 3)
z.preview()
z.save("sic.mp4")

