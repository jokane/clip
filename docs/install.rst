============
Installation
============

You can install ``clip`` directly from github::

    $ pip install -U git+https://github.com/jokane/clip

Or to make an executable script using ``uv``, you can start with something
like this::

    #!/usr/bin/env -S uv run --script
    # /// script
    # dependencies = ["clip @ git+https://github.com/jokane/clip"]
    # ///

    import clip

To use ``clip`` to read and write video files, you must also install ``ffmpeg``
separately::

    $ sudo apt install ffmpeg

