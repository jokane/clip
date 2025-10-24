"""

An example that creates two videos: one that shows a block of text bouncing
around the screen and one with some mesmerizing colors.

"""

import bisect
import math

import scipy.ndimage
import numpy as np

import clip

class BouncePositioner:
    """A class whose instances can be used as the position parameter when
    creating an Element to use in a composite.  Simulates movement within the
    given boundaries at the given velocity, bouncing off the edges as
    needed.

    :param width: The integer width of the rectangle in which to bounce, in
            pixels.
    :param width: The integer height of the rectangle in which to bounce, in
            pixels.
    :param start_position: A 2-tuple giving the starting position.
    :param start_velocity: A 2-tuple giving the initial velocity.  This will
            change only when the object reaches the boundary and bounces away.

    """
    def __init__(self, width, height, start_position, start_velocity):
        self.width = width
        self.height = height

        position = np.array(start_position)
        velocity = np.array(start_velocity)

        t = np.float64(0)

        self.bounces = [(t, position, velocity)]

    def compute_bounce(self, t, position, velocity):
        """ If the object is at the given position at the given time, moving
        with the given velocity, compute and return the time, position, and new
        velocity when it bounces next."""

        dt = float('inf')
        side = None

        if velocity[0] > 0:
            dt_right = (self.width - position[0])/velocity[0]
            if dt_right < dt:
                side = 'right'
                dt = dt_right

        if velocity[0] < 0:
            dt_left = -position[0]/velocity[0]
            if dt_left < dt:
                side = 'left'
                dt = dt_left

        if velocity[1] > 0:
            dt_bottom = (self.height - position[1])/velocity[1]
            if dt_bottom < dt:
                side = 'bottom'
                dt = dt_bottom

        if velocity[1] < 0:
            dt_top = -position[1]/velocity[1]
            if dt_top < dt:
                side = 'top'
                dt = dt_top

        new_position = position + dt * velocity
        new_t = t + dt

        if side in ('left', 'right'):
            new_velocity = np.array([-velocity[0], velocity[1]])
        else:
            new_velocity = np.array([velocity[0], -velocity[1]])

        return new_t, new_position, new_velocity

    def __call__(self, t):
        """Return the position of the object at time t as a tuple of
        integers."""

        # Make sure we have computed enough bounces to get out to the requested
        # time.
        while t > self.bounces[-1][0]:
            new_bounce = self.compute_bounce(*self.bounces[-1])
            self.bounces.append(new_bounce)

        # Find the pair of bounces that we are moving between and extract the
        # time and position for each one.
        index = bisect.bisect_left(self.bounces, np.float64(t), key=lambda x: x[0])
        b1, b2 = self.bounces[index-1], self.bounces[index]
        t1, t2 = b1[0], b2[0]
        p1, p2 = b1[1], b2[1]

        # Compute the exact position between these two bounces.
        alpha = (t2 - t)/(t2 - t1)
        p = alpha*p1 + (1-alpha)*p2
        int_p = list(map(int, p))

        # Done!
        return int_p


class color_gradient(clip.VideoClip):
    """A clip showing a given color, with transparency decreasing with distance
    from a given point, which may be moving.

    :param color: The color to fill the frames of the clip.
    :param position: Which poiny should be fully opaque?  Either an integer
            point, or a callable that maps times to integer points.
    :param width: The positive integer width of the clip.
    :param height: The positive integer height of the clip.
    :param length: The positive float length of the clip.

    """

    def __init__(self, color, position, width, height, length):
        super().__init__()

        self.color = color
        self.position = position
        self.metrics = clip.Metrics(src=clip.Clip.default_metrics,
                                    width=width,
                                    height=height,
                                    length=length)

    def get_position(self, t):
        """Where will the object be at time t?"""
        if callable(self.position):
            return self.position(t)
        else:
            return self.position

    def frame_signature(self, t):
        return ['color_gradient',
                self.color,
                self.get_position(t),
                self.width(),
                self.height()]

    def request_frame(self, t):
        pass

    def get_frame(self, t):
        width = self.width()
        height = self.height()
        diam = math.sqrt(math.sqrt(height*height + width*width))

        pos = list(map(int, self.get_position(t)))

        x = 1 - np.zeros((height, width))
        x[pos[1],pos[0]] = 0
        x = scipy.ndimage.distance_transform_edt(x)
        x = np.sqrt(x)
        x = 255 - (255.0/diam)*x
        x = x.astype(np.uint8)
        c = np.full((height, width, 3),
                    (self.color[2], self.color[1], self.color[0]),
                    dtype=np.uint8)
        x = np.stack([c[:,:,0], c[:,:,1], c[:,:,2], x], axis=2)
        return x

def video1():
    """Make a video with some text bouncing around."""
    screen_width, screen_height = 640, 480
    length = 60

    font_filename="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    text_clip1 = clip.draw_text(text="DVD", font_filename=font_filename,
                                font_size=50, color=(255,255,255), length=length)

    text_clip2 = clip.draw_text(text="Video", font_filename=font_filename,
                                font_size=50, color=(255,255,255), length=length)

    text_clip = clip.vstack(text_clip1, 5, text_clip2)

    bp = BouncePositioner(width=screen_width-text_clip.width(),
                           height=screen_height-text_clip.height(),
                           start_position=(100, 100),
                           start_velocity=(100, 100))

    video = clip.composite(clip.Element(text_clip, 0, bp))

    clip.save_mp4(video, 'text_bounce.mp4', 30)

def video2():
    """Make a video with some shifting colors."""
    width = 960
    height = 540

    elements = []
    layers = [{'color': (0, 0, 255),
               'start': (width/2, 0),
               'velocity': (100, 12.5),
               'alpha_scale': 1.0},
              {'color': (255, 0, 0),
               'start': (0, height/2),
               'velocity': (100, -120.5),
               'alpha_scale': 0.5},
              {'color': (0, 255, 0),
               'start': (width/2, height/2),
               'velocity': (-100, -120.5),
               'alpha_scale': 0.333},
              {'color': (0, 255, 255),
               'start': (width, height),
               'velocity': (-100, -100),
               'alpha_scale': 0.25}
              ]
    for layer in layers:
        bp = BouncePositioner(width=width,
                              height=height,
                              start_position=layer['start'],
                              start_velocity=layer['velocity'])

        x = color_gradient(color=layer['color'],
                           position=bp,
                           width=width,
                           height=height,
                           length=60)

        x = clip.scale_alpha(x, layer['alpha_scale'])

        e = clip.Element(clip=x,
                    start_time=0,
                    position=(0,0),
                    video_mode=clip.VideoMode.BLEND)

        elements.append(e)

    x = clip.composite(elements)

    clip.save_mp4(x, 'colors.mp4', 30)

if __name__ == '__main__':
    video1()
    video2()

