"""
An example that shows a way to reduce noise in the audio of a clip.
"""
import os
import urllib.request

import noisereduce
import numpy as np

import clip

class reduce_noise(clip.MutatorClip):
    """Attempt to reduce the noise in the audio of a clip, based on a snippet
    whose audio is known to be the same kind of noise.  Video is unchanged."""
    def __init__(self, signal):
        super().__init__(signal)
        self.samples = None

    def get_samples(self):
        if self.samples is None:
            signal_data = self.clip.get_samples()
            self.samples = np.zeros(signal_data.shape, dtype=signal_data.dtype)

            for channel in range(self.clip.num_channels()):
                print(f"[reducing noise in {self.num_samples()} samples, "
                      f"channel {channel+1} of {self.num_channels()}]")
                self.samples[:,channel] = noisereduce.reduce_noise(signal_data[:,channel],
                                                                   self.sample_rate())

        return self.samples

def main():
    """Make a short video based on a famous speech."""

    # Grab a video of a famous speech.
    url = "https://ia600209.us.archive.org/11/items/John-F-Kennedy_Speech_Rice-Stadium/JFK%20Adress%20at%20Rice%20University_720p.mp4" # pylint: disable=line-too-long
    filename = '1962-09-12.mp4'

    if not os.path.exists(filename):
        print(f'Downloading {url}...')
        urllib.request.urlretrieve(url, filename)

    # Extract the relevant sound bite.
    #
    # Aside: To get the whole speech instead of just this paragraph, try
    # start=14*60+15 and end=32*60+5 instead.
    video = clip.from_file(filename)
    soundbite = clip.slice_clip(video, start=22*60+39, end=23*60+15)

    # Reduce the audio noise.
    cleaned_soundbite = reduce_noise(soundbite)

    # Alternate between the original and the cleaned one, to make the
    # difference easier to hear.
    step = 3
    current_t = 0
    current_clip, other_clip = clip.to_monochrome(soundbite), cleaned_soundbite
    elements = []
    while current_t < soundbite.length():
        next_t = min(current_t+step, soundbite.length())
        element = clip.Element(clip.slice_clip(current_clip, current_t, next_t), current_t, (0, 0))
        elements.append(element)
        current_clip, other_clip = other_clip, current_clip
        current_t = next_t

    alternating = clip.composite(elements)

    # Save just the audio.
    clip.save_audio(cleaned_soundbite, 'jfk-why-moon.flac')

    # Save the full cleaned result including video.
    clip.save_mp4(cleaned_soundbite, 'jfk-why-moon-clean.mp4', video.frame_rate)

    # Save the alternating version with video.
    clip.save_mp4(alternating, 'jfk-why-moon-alternating.mp4', video.frame_rate)



if __name__ == '__main__':
    main()
