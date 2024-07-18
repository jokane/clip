====
Clip
====

This is a Python library for creating videos.

To extract the first minute of an existing video, you might do something like this::

    from clip import *
    original = from_file('original.mp4')
    first_minute = slice_clip(original, 0, 60)
    save_mp4(first_minute, 'first_minute.mp4', frame_rate=original.frame_rate)

Loads of other functionality is included.

- Import different sorts of source material, including
  `most video and audio files readable by FFMPEG <https://jokane.github.io/clip/reference.html#clip.from_file>`_,
  `static images <https://jokane.github.io/clip/reference.html#clip.static_image>`_,
  `zipped archives of image sequences <https://jokane.github.io/clip/reference.html#clip.zip_file>`_,
  `PDF documents <https://jokane.github.io/clip/reference.html#clip.pdf_page>`_, and
  `rosbags <https://jokane.github.io/clip/reference.html#clip.from_rosbag>`_.
 
- Combine source material by
  `sequencing multiple clips <https://jokane.github.io/clip/reference.html#clip.chain>`_,
  stacking multiple simultaneous clips `horizontally <https://jokane.github.io/clip/reference.html#clip.hstack>`_
  or `vertically <https://jokane.github.io/clip/reference.html#clip.vstack>`_, or
  `other customized composite operations <https://jokane.github.io/clip/reference.html#clip.composite>`_.

- `Add text <https://jokane.github.io/clip/reference.html#clip.draw_text>`_,
  `fade in <https://jokane.github.io/clip/reference.html#clip.fade_in>`_,
  `fade out <https://jokane.github.io/clip/reference.html#clip.fade_out>`_,
  or apply
  `other filters <https://jokane.github.io/clip/reference.html#clip.filter_frames>`_.

`This video <https://jokane.net/pubs/MoaSheOKa21b.mp4>`_ is an example of many
of the things the library can do.

See the `documentation <https://jokane.github.io/clip>`_ for more details.

