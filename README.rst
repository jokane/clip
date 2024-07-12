====
Clip
====

This is a Python library for creating videos.

To extract the first minute of an existing video, you might do something like this::

    from clip import *
    original = from_file('original.mp4')
    first_minute = slice_clip(original, 0, 60)
    first_minute.save('first_minute.mp4', frame_rate=original.frame_rate)

Loads of other functionality is included.

- Import different sorts of source material, including
  `most video and audio files readable by FFMPEG <https://jokane.github.io/clip/_user/from_file.html>`_,
  `static images <https://jokane.github.io/clip/_user/static_image.html>`_,
  `zipped archives of image sequences <https://jokane.github.io/clip/_user/zip_file.html>`_,
  `PDF documents <https://jokane.github.io/clip/_user/pdf_page.html>`_, and
  `rosbags <https://jokane.github.io/clip/_user/from_rosbag.html>`_.
 
- Combine source material by
  `sequencing multiple clips <https://jokane.github.io/clip/_user/chain.html>`_,
  stacking multiple simultaneous clips `horizontally <https://jokane.github.io/clip/_user/hstack.html>`_ or `vertically <https://jokane.github.io/clip/_user/vstack.html>`_, or
  `other customized composite operations <https://jokane.github.io/clip/_user/composite.html>`_.

- `Add text <https://jokane.github.io/clip/_user/draw_text.html>`_,
  `fade in <https://jokane.github.io/clip/_user/fade_in.html>`_,
  `fade out <https://jokane.github.io/clip/_user/fade_out.html>`_,
  or apply
  `other filters <https://jokane.github.io/clip/_user/zip_file.html>`_.

`This video <https://jokane.net/pubs/MoaSheOKa21b.mp4>`_ is an example of many
of the things the library can do.

See the `documentation <https://jokane.github.io/clip>`_ for more details.

