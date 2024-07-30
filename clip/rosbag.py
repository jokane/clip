""" A tool to treat a topic rosbag ---i.e. recorded data from the ROS robotics
framework--- as a video. """


import bisect
import itertools
import os
import pathlib
import statistics

import cv2

import rosbags.highlevel
import rosbags.typesys
import rosbags.typesys.stores.ros2_humble
from rosbags.typesys.stores.ros2_humble import (builtin_interfaces__msg__Time as Time,
                                              sensor_msgs__msg__CompressedImage as CompressedImage,
                                              std_msgs__msg__Header as Header)

from .base import frame_times, Clip, VideoClip
from .progress import custom_progressbar
from .metrics import Metrics

class ROSImageMessage():
    """A compressed image message, read from a rosbag.  Used by :class:`from_rosbag`.

    :param reader: A `Reader` object from the `rosbags` package.
    :param tup: A tuple `(connection, topic, rawdata)` supplied by `reader`.

    """
    def __init__(self, reader, tup):
        # Ignoring the timestamp stored in the rosbag, because the one in the
        # message header should be more accurate.
        connection, _, rawdata = tup

        self.topic = connection.topic
        self.message = reader.deserialize(rawdata, connection.msgtype)
        stamp = self.message.header.stamp
        self.timestamp = stamp.sec + stamp.nanosec/1e9

    def image(self):
        """Decompress and return the image encoded in this message."""
        frame = cv2.imdecode(self.message.data, cv2.IMREAD_COLOR)
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        return frame


class from_rosbag(VideoClip):
    """Read images from given topic in a rosbag and treat them as a silent video. |from-source|

    :param pathname: The name of a ROS1 bag file or a ROS2 bag directory.

    :param topic: The name of a topic in the bag file, of type
            `sensor_msgs/CompressedImage`.

    .. automethod:: from_rosbag.estimated_frame_rate

    """
    def __init__(self, pathname, topic):
        super().__init__()
        self.pathname = pathname
        self.timestamp = os.path.getmtime(self.pathname)
        self.topic = topic

        path = pathlib.Path(pathname)
        typestore = rosbags.typesys.get_typestore(rosbags.typesys.Stores.ROS2_HUMBLE)
        reader = rosbags.highlevel.AnyReader(paths=[path],
                                             default_typestore=typestore)
        reader.open()
        connections = [x for x in reader.connections if x.topic == topic]
        self.messages = list(map(lambda x: ROSImageMessage(reader, x),
                                 reader.messages(connections=connections)))

        self.sample_frame = self.messages[0].image()
        height, width, _ = self.sample_frame.shape

        l = self.messages[-1].timestamp - self.messages[0].timestamp

        self.metrics = Metrics(src=Clip.default_metrics,
                               width = width,
                               height = height,
                               length = l)

    def index_of(self, t):
        """Return the index of the message that should be displayed at the given time."""
        target_stamp = self.messages[0].timestamp + t
        return bisect.bisect_left(self.messages, target_stamp, key=lambda msg: msg.timestamp) - 1

    def estimated_frame_rate(self):
        """Return an estimate of the native frame rate of this image sequence,
        based on the median time gap between successive frames."""
        gaps = [b.timestamp - a.timestamp for a, b in itertools.pairwise(self.messages)]
        return 1.0/statistics.median(gaps)

    def frame_signature(self, t):
        return [ 'rosbag',
                self.pathname,
                self.timestamp,
                self.topic,
                t ]

    def get_frame(self, t):
        i = self.index_of(t)
        return self.messages[i].image()

    def request_frame(self, t):
        pass

def save_rosbag(clip, pathname, frame_rate, topic='/camera/compressed', frame_id='/camera'):
    """Save the video portion of a clip as a ROS2 rosbag, in a topic of type
    `sensor_msgs/CompressedImage`. |save|

    :param pathname: The name of the directory to write to.
    :param frame_rate: The desired frame rate.
    :param topic: The topic name.
    :param frame_id: The frame_id to use in the message headers.

    """

    path = pathlib.Path(pathname)
    typestore = rosbags.typesys.get_typestore(rosbags.typesys.Stores.ROS2_HUMBLE)

    with custom_progressbar(f"Saving {path}", round(clip.length(), 1)) as pb:
        with rosbags.rosbag2.Writer(path, version=8) as writer:
            msgtype = CompressedImage.__msgtype__
            connection = writer.add_connection(topic=topic,
                                               msgtype=msgtype,
                                               typestore=typestore)

            for t in frame_times(clip.length(), frame_rate):
                pb.update(round(t, 1))

                t_in_nanos = int(t*1e9)

                stamp = Time(sec=int(t), nanosec=int((t-int(t))*1e9))

                header = Header(stamp = stamp,
                                frame_id=frame_id)

                frame = clip.get_frame(t)
                frame_compressed = cv2.imencode('.jpg', frame)[1]

                msg = CompressedImage(format='rgb8; jpeg compressed bgr8',
                                      header=header,
                                      data=frame_compressed)

                msg_serialized = typestore.serialize_cdr(msg, msgtype)

                writer.write(connection,
                             t_in_nanos,
                             msg_serialized)

