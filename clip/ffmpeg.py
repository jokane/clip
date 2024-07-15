""" A function for running ffmpeg with a given set of arguments.  Tracks
progress by watching the stats file generated along the way. """

import os
import shutil
import subprocess
import re
import tempfile
import time
import threading

from .cache import ClipCache
from .progress import custom_progressbar
from .util import temporarily_changed_directory

def save_via_ffmpeg(clip, filename, frame_rate, output_args, cache_dir, two_pass):
    """Use ffmpeg to save a clip with the given arguments describing the
    desired output.

    :param clip: The clip to save.
    :param filename: A file name to write the video to.
    :param frame_rate: The frame rate to use for the final output.
    :param output_args: A list of string arguments to pass to `ffmpeg`.  These
        will appear after arguments setting up the inputs.
    :param two_pass: Should `ffmpeg` run twice or just once?
    """

    # Construct the complete path name. We'll need this during the ffmpeg step,
    # because that runs in a temporary current directory.
    full_fname = os.path.join(os.getcwd(), filename)

    # Read the frame cache.  This needs to happen before we change to the temp
    # directory.
    cache = ClipCache(cache_dir)
    cache.scan_directory()

    # Compute the number of frames in the output, so ffmpeg() can use it for
    # its progress bar.
    num_frames = int(clip.length() * frame_rate)

    # Do the real work in a temporary directory, so we don't make a mess.
    with tempfile.TemporaryDirectory() as td:
        # Fill the temporary directory with the audio, the subtitles, and a
        # bunch of (symlinks to) individual frames.
        clip.stage(directory=td,
                   cache=cache,
                   frame_rate=frame_rate,
                   fname=filename)

        # In the temporary directory, invoke ffmpeg to assemble the completed
        # video.
        with temporarily_changed_directory(td):
            # Assemble arguments that describe the inputs to ffmpeg.
            input_args = []
            input_args.append(f'-framerate {frame_rate}')
            input_args.append(f'-i %06d.{cache.frame_format}')
            input_args.append('-i audio.flac')
            if os.stat('subtitles.srt').st_size > 0:
                input_args.append('-i subtitles.srt -c:s mov_text -metadata:s:s:0 language=eng')

            if not two_pass:
                ffmpeg(task=f"Encoding {filename}",
                       *(input_args + output_args + [f'{full_fname}']),
                       num_frames=num_frames)
            else:
                ffmpeg(task=f"Encoding {filename}, pass 1",
                       *(input_args + output_args + ['-pass 1', '/dev/null']),
                       num_frames=num_frames)
                ffmpeg(task=f"Encoding {filename}, pass 2",
                       *(input_args + output_args + ['-pass 2', f'{full_fname}']),
                       num_frames=num_frames)

    print(f'Wrote {clip.readable_length()} to {filename}.')

class FFMPEGException(Exception):
    """Raised when ffmpeg fails for some reason."""

def ffmpeg(*args, task=None, num_frames=None):
    """Run ffmpeg with the given arguments.  Optionally, maintain a progress
    bar as it goes."""

    with tempfile.NamedTemporaryFile() as stats:
        command = f"ffmpeg -y -vstats_file {stats.name} {' '.join(args)} 2> errors"
        with subprocess.Popen(command, shell=True) as proc:
            t = threading.Thread(target=proc.communicate)
            t.start()

            if task is not None:
                with custom_progressbar(task=task, steps=num_frames) as pb:
                    pb.update(0)
                    while proc.poll() is None:
                        try:
                            with open(stats.name) as f: #pragma: no cover
                                fr = int(re.findall(r'frame=\s*(\d+)\s', f.read())[-1])
                                pb.update(min(fr, num_frames-1))
                        except FileNotFoundError:
                            pass # pragma: no cover
                        except IndexError:
                            pass # pragma: no cover
                        time.sleep(1)

            t.join()

            if os.path.exists('errors'):
                shutil.copy('errors', '/tmp/ffmpeg_errors')
                with open('/tmp/ffmpeg_command', 'w') as f:
                    print(command, file=f)

            if proc.returncode != 0:
                if os.path.exists('errors'):
                    with open('errors', 'r') as f:
                        errors = f.read()
                else:
                    errors = '[no errors file found]' #pragma: no cover
                message = (
                  f'Alas, ffmpeg failed with return code {proc.returncode}.\n'
                  f'Command was: {command}\n'
                  f'Standard error was:\n{errors}'
                )
                raise FFMPEGException(message)
