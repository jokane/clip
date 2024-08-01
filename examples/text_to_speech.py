"""

An example that uses a text-to-speech library to generate spoken word from a
string input.

Install `pyttsx4` before running:

        pip install pyttsx4

"""
import io
import re
import textwrap

import pyttsx4
import soundfile

import clip


class text_to_speech(clip.from_audio_samples):
    """A clip of the given string spoken aloud, including both audio and
    subtitles.

    :param text: The text to be spoken.  A string.
    :param words_per_minute: The requested number of words per minute to speak.
            A positive integer, or `None` to use the library default which is
            apparently 200.

    """

    def __init__(self, text, words_per_minute=None):
        self.text = text

        engine = pyttsx4.init()
        if words_per_minute is not None:
            engine.setProperty('rate', words_per_minute)
        bio = io.BytesIO()
        engine.save_to_file(text, bio)
        engine.runAndWait()

        bio.seek(0)

        sample_rate = 22050
        num_channels = 1

        samples = soundfile.read(bio,
                                 channels=num_channels,
                                 samplerate=sample_rate,
                                 format='RAW',
                                 subtype='PCM_16',
                                 always_2d=True)[0]

        super().__init__(samples, sample_rate)

    def get_subtitles(self):
        """ Generate the subtitles automatically from the given text.  Estimate
        their timing based on a constant number of characters spoken per second.
        These timings are not perfect, but generally seem not to get too far
        off."""
        text = re.sub(r'\s+', ' ', self.text).strip()
        total_seconds = self.length()
        total_characters = len(self.text)
        seconds_so_far = 0

        for chunk in textwrap.wrap(text, width=35):
            characters_this_chunk = len(chunk) + 1
            seconds_this_chunk = total_seconds * characters_this_chunk/ total_characters
            seconds_after_this_chunk = seconds_so_far + seconds_this_chunk
            yield (seconds_so_far, seconds_after_this_chunk, chunk)
            seconds_so_far = seconds_after_this_chunk

def main():
    message = """
    Four score and seven years ago our fathers brought forth on
    this continent a new nation, conceived in liberty, and dedicated to the
    proposition that all men are created equal. Now we are engaged in a great
    civil war, testing whether that nation, or any nation so conceived and so
    dedicated, can long endure. We are met on a great battlefield of that war.
    We have come to dedicate a portion of that field, as a final resting place
    for those who here gave their lives that that nation might live. It is
    altogether fitting and proper that we should do this. But, in a larger
    sense, we cannot dedicate, we cannot consecrate, we cannot hallow this
    ground. The brave men, living and dead, who struggled here, have
    consecrated it, far above our poor power to add or detract. The world will
    little note, nor long remember what we say here, but it can never forget
    what they did here. It is for us the living, rather, to be dedicated here
    to the unfinished work which they who fought here have thus far so nobly
    advanced. It is rather for us to be here dedicated to the great task
    remaining before us that from these honored dead we take increased devotion
    to that cause for which they gave the last full measure of devotion that we
    here highly resolve that these dead shall not have died in vain that this
    nation, under God, shall have a new birth of freedom and that government of
    the people, by the people, for the people, shall not perish from the earth.
    """

    x = text_to_speech(message, words_per_minute=150)

    font_filename="/usr/share/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"

    stack_items = []
    stack_items.append(100)
    for line in message.strip().split('\n'):
        text_clip = clip.draw_text(text="     " + line.strip() + "     ",
                                   font_filename=font_filename,
                                   font_size=25,
                                   color=(255,255,255),
                                   length=x.length())
        stack_items.append(text_clip)
        stack_items.append(int(0.2*text_clip.height()))
    stack_items.append(100)
    y = clip.vstack(stack_items, align=clip.Align.CENTER)

    z = clip.background(clip.join(y, x), (128, 0, 0))
    clip.save_mp4(z, 'address.mp4', frame_rate=30)

if __name__ == '__main__':
    main()
