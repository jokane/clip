""" A function for running ffmpeg with a given set of arguments.  Tracks
progress by watching the stats file generated along the way. """

import os
import shutil
import subprocess
import re
import tempfile
import time
import threading

from .progress import custom_progressbar

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
