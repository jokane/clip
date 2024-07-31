"""

An example that shows a block of text bouncing around the screen.

"""

import bisect

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

        # Where are the places we might bounce? Each element of this list is a
        # tuple: (name, which dimension?, what limit in that dimension?)
        limits = [(0, 0),
                  (0, self.width),
                  (1, 0),
                  (1, self.height)]

        # For each limit, how long will it take to get there?  Add the time at
        # the front of the tuple.
        limits_with_times = [((lim - position[dim])/velocity[dim], dim, lim)
                                 for dim, lim in limits ]

        # Reject times that not positive -- those represent limits that we're
        # moving away from, and thus will never reach without bouncing at some
        # point.
        active_limits = [ x for x in limits_with_times if x[0] > 0 ]

        # Which of these remaining limts will we reach on the next bounce?  We
        # can use the standard ordering to find the minimum because the times
        # are the first elements of each limit tuple.  But it might be more
        # that one, if we're hitting the corner exactly.
        dt = min(active_limits)[0]
        next_limits = [ x for x in active_limits if x[0] == dt ]

        # Compute resulting position and the new velocity after the bounce.
        dt, dim, lim = next_limits[0]
        new_position = position + dt * velocity
        new_velocity = velocity.copy()
        t += dt
        for _, _, dim, _ in next_limits:
            new_velocity[dim] *= -1

        return t, new_position, new_velocity

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

if __name__ == '__main__':
    screen_width, screen_height = 640, 480
    length = 60

    font_filename="/usr/share/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"

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

    clip.save_mp4(video, 'test.mp4', 30)

