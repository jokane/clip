from clip import MutatorClip, require_clip, Metrics, solid, save_mp4, VideoClip, Clip, composite, Element, VideoMode, draw_text, black, superimpose_center, hstack, filter_frames, ken_burns, hold_at_start, hold_at_end

import numpy as np

class replace_alpha(MutatorClip):
    def __init__(self, clip, alpha_clip):
        super().__init__(clip) 
        require_clip(alpha_clip, 'alpha clip')
        self.alpha_clip = alpha_clip

    def frame_signature(self, t):
        return ['replace_alpha',
                 self.clip.frame_signature(t),
                 self.alpha_clip.frame_signature(t)]

    def get_frame(self, t):
        frame = self.clip.get_frame(t).copy()
        alpha_frame = self.alpha_clip.get_frame(t)
        frame[:,:,3] = alpha_frame[:,:,3]
        return frame

class wipe(VideoClip):
    def __init__(self, scale, width, height, length):
        super().__init__()
        self.metrics = Metrics(Clip.default_metrics,
                               width=width,
                               height=height,
                               length=length)
        self.scale = scale

    def frame_signature(self, t):
        return [ 'wipe', self.metrics, self.scale, t ]

    def request_frame(self, t):
        pass

    def get_frame(self, t):
        start = 2 * t / self.metrics.length * self.scale - self.scale
        stop = start - self.scale
        x = np.linspace(start=start,
                        stop=stop,
                        num = self.metrics.width)

        x = 1/(1 + np.exp(-x))
        x *= 255
        x = x.astype(np.uint8)
        x = np.array([x] * self.metrics.height)

        frame = np.zeros([self.metrics.height, self.metrics.width, 4], np.uint8)
        frame[:,:,3] = x
        
        return frame

def reflect_horizontal(clip):
    return filter_frames(clip, lambda frame: np.flip(frame, axis=1))

if __name__ == '__main__':
    length = 10
    scale = 3
    font_size=120

    font_filename="/usr/share/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"

    def wiped_text(text, scale, flip):
        text_clip = draw_text(text=text,
                              font_filename=font_filename,
                              font_size=font_size,
                              color=(255,255,255),
                              length=length)
    
        wipe_alpha = wipe(scale=scale,
                          width=text_clip.width(),
                          height=text_clip.height(),
                          length=text_clip.length())
        if flip:
            wipe_alpha = reflect_horizontal(wipe_alpha)

        wiped = replace_alpha(clip=text_clip,
                              alpha_clip=wipe_alpha)

        return wiped

        
    text = hstack(wiped_text('hello', scale, True),
                  int(font_size/2),
                  wiped_text('world', scale, False))

    background = black(800, 600, length)
    complete = superimpose_center(background, text, 0, VideoMode.BLEND)

    held = hold_at_end(hold_at_start(complete, length+1), length+4)

    zoomed = ken_burns(clip=held,
                       width=800,
                       height=600,
                       start_top_left=(0,0),
                       start_bottom_right=(640, 480),
                       end_top_left=(0,0),
                       end_bottom_right=(800, 600))

    save_mp4(clip=zoomed,
             filename='hi.mp4',
             frame_rate=30)

