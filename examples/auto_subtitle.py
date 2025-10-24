#!/usr/bin/env python
"""

An example that uses a speech-to-text library to try to add subtitles to a clip
automatically.  This example uses `vosk` because it works offline and provides
easy access to timing of individual words.

Install `vosk` before running::

        pip install vosk


"""


import json
import os
import urllib.request

import numpy as np
import vosk

import clip


class auto_subtitle(clip.MutatorClip):
    """ Replace the subtiles in the given clip with auto-generated subtitles
    based on speech recognition.

    :param clip: The clip to modify.
    :param max_characteres: The maximum number of characters in a single
            subtitle.
    :param max_pause: The longest allowable pause between words that appear in
            a single subtitle.  When there's a pause longer than this, leave a
            subtitle-free gap.

    """
    def __init__(self, clip_, max_characters=35, max_pause=0.75):
        super().__init__(clip_)

        if self.clip.num_channels() == 1:
            self.mono_clip = self.clip
        else:
            self.mono_clip = clip.stereo_to_mono(self.clip)

        self.max_characters = max_characters
        self.max_pause = max_pause

    def get_subtitles(self):
        audio_array = self.mono_clip.get_samples()
        audio_bytes = (32767*audio_array).astype(np.int16).tobytes()

        def chunks(data, stride):
            i = 0
            while i < len(data):
                yield data[i:i+stride]
                i += stride

        vosk.SetLogLevel(-1)
        model = vosk.Model(lang="en-us")
        rec = vosk.KaldiRecognizer(model, self.clip.sample_rate())
        rec.SetWords(True)
        rec.SetPartialWords(True)

        with clip.custom_progressbar('Auto-captioning', round(self.length()+0.5)) as pb:
            for t, chunk in enumerate(chunks(audio_bytes, 2*self.clip.sample_rate())):
                rec.AcceptWaveform(chunk)
                pb.update(t)

        result = json.loads(rec.FinalResult())

        if 'result' not in result:
            raise ValueError('Speech recognition failed.')

        def group_words(words):
            current_start = None
            current_end = None
            current_text = ""

            for word in words:
                stop = (len(current_text) + len(word['word']) > self.max_characters or
                    current_end and word['start'] - current_end > self.max_pause)

                if stop:
                    yield((current_start, current_end, current_text))
                    current_start = None
                    current_end = None
                    current_text = ''

                if current_start is None:
                    current_start = word['start']
                    current_end = word['end']
                    current_text = word['word']
                else:
                    current_end = word['end']
                    current_text += ' ' + word['word']

        yield from group_words(result['result'])


def main():
    """Make a short video based on a famous speech."""

    # pylint: disable=line-too-long
    url = "https://www.fdrlibrary.org/documents/356632/405112/afdr244.mp3/b37c9e47-9056-4932-b08b-d85cc22e586b"
    filename = '1941-12-07.mp3'

    if not os.path.exists(filename):
        print(f'Downloading {url}...')
        urllib.request.urlretrieve(url, filename)

    speech = clip.from_file(filename)
    speech_sliced = clip.slice_clip(speech, start=800, end=1212)
    speech_sliced_subtitled = auto_subtitle(speech_sliced)

    clip.save_mp4(speech_sliced_subtitled, 'roosevelt.mp4', 5)

if __name__ == '__main__':
    main()
