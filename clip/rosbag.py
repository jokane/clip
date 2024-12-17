""" A tool to treat a topic rosbag ---i.e. recorded data from the ROS robotics
framework--- as a video. """


import bisect
import itertools
import os
import pathlib
import statistics

import numpy as np

import cv2

import rosbags.highlevel
import rosbags.typesys
import rosbags.typesys.stores.ros2_humble
from rosbags.typesys.stores.ros2_humble import (builtin_interfaces__msg__Time as Time,
                                              sensor_msgs__msg__CompressedImage as CompressedImage,
                                              sensor_msgs__msg__Image as Image,
                                              std_msgs__msg__Header as Header)

from .base import frame_times, Clip, VideoClip, require_bool
from .progress import custom_progressbar
from .metrics import Metrics

class ROSImageMessage():
    """An image message, read from a rosbag.  Used by :class:`from_rosbag`.

    :param reader: A `Reader` object from the `rosbags` package.
    :param tup: A tuple `(connection, topic, rawdata)` supplied by `reader`.

    """
    def __init__(self, reader, tup):
        # Ignoring the timestamp stored in the rosbag, because the one in the
        # message header should be more accurate.
        connection, _, rawdata = tup

        self.topic = connection.topic
        self.msgtype = connection.msgtype
        self.message = reader.deserialize(rawdata, connection.msgtype)
        stamp = self.message.header.stamp
        self.timestamp = stamp.sec + stamp.nanosec/1e9

    def image(self):
        """Decode, possibly decompress, and return the image encoded in this
        message."""
        if self.msgtype == 'sensor_msgs/msg/Image':
            if self.message.encoding == 'mono8':
                mono_frame = np.reshape(self.message.data,
                                        (self.message.height, self.message.width))
                frame = cv2.cvtColor(mono_frame, cv2.COLOR_GRAY2RGBA)
            elif self.message.encoding == 'rgb8':
                rgb_frame = np.reshape(self.message.data,
                                       (self.message.height, self.message.width, 3))
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2RGBA)

            else:
                raise ValueError('Unsupported uncompressed image encoding: '
                                 f'{self.message.encoding}') #pragma: nocover
        elif self.msgtype == 'sensor_msgs/msg/CompressedImage':
            frame = cv2.imdecode(self.message.data, cv2.IMREAD_COLOR)
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        else:
            raise ValueError(f'Unknown message type: {self.msgtype}') #pragma: nocover

        return frame


class from_rosbag(VideoClip):
    """Read images from given topic in a rosbag and treat them as a silent
    video. |from-source|

    :param pathname: The name of a ROS1 bag file or a ROS2 bag directory.

    :param topic: The name of a topic in the bag file, of type
            `sensor_msgs/msg/Image` or
            `sensor_msgs/msg/CompressedImage`.

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

        if len(connections) == 0:
            raise ValueError(f'In rosbag {pathname}, topic {topic} does not exist.')

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
        """Return the index of the message that should be displayed at the
        given time."""
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

def save_rosbag(clip, pathname, frame_rate, compressed=True,
                topic=None, frame_id='/camera', fmt=None):
    """Save the video portion of a clip as a ROS2 rosbag. |save|

    :param pathname: The name of the directory to write to.
    :param frame_rate: The desired frame rate.
    :param compressed: Boolean telling whether the images should be compressed.
    :param topic: The topic name.
    :param frame_id: The frame_id to use in the message headers.
    :param fmt: The format or encoding to use for the images.

    If `compressed` is `True`:

    * The topic will default to `/camera/image_raw/compressed`.
    * The topic's data type will be `sensor_msgs/CompressedImage`.
    * The `format` parameter will refer to the encoding of the images,
      defaulting to `rgb8`.

    If `compressed` is `False`:

    * The topic will default to `/camera/image_raw`.
    * The topic's data type will be `sensor_msgs/CompressedImage`.
    * The `format` parameter will refer to the compression format, defaulting
      to `rgb8; jpeg compressed bgr8`.

    """

    path = pathlib.Path(pathname)
    typestore = rosbags.typesys.get_typestore(rosbags.typesys.Stores.ROS2_HUMBLE)

    require_bool(compressed, "compressed")

    if compressed:
        msgtype = CompressedImage.__msgtype__
        if topic is None:
            topic = '/camera/image_raw/compressed'
        if fmt is None:
            fmt = 'rgb8; jpeg compressed bgr8'
    else:
        msgtype = Image.__msgtype__
        if topic is None:
            topic = '/camera/image_raw'
        if fmt is None:
            fmt = 'rgb8'

    clip.request_all_frames(frame_rate)

    with custom_progressbar(f"Saving {path}", round(clip.length(), 1)) as pb:
        with rosbags.rosbag2.Writer(path, version=8) as writer:

            connection = writer.add_connection(topic=topic,
                                               msgtype=msgtype,
                                               typestore=typestore)

            for t in frame_times(clip.length(), frame_rate):
                pb.update(round(t, 1))

                t_in_nanos = int(t*1e9)

                stamp = Time(sec=int(t), nanosec=int((t-int(t))*1e9))

                header = Header(stamp = stamp,
                                frame_id=frame_id)

                frame_rgba = clip.get_frame(t)

                if compressed:
                    frame_compressed = cv2.imencode('.jpg', frame_rgba)[1]
                    msg = CompressedImage(header=header,
                                          format=fmt,
                                          data=frame_compressed)
                else:
                    if fmt == 'rgb8':
                        step=clip.width()*3
                        frame_rgb = frame_rgba[:,:,:3]
                        data=frame_rgb.flatten()
                    elif fmt == 'mono8':
                        step=clip.width()
                        frame_mono = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2GRAY)
                        data=frame_mono.flatten()
                    else:
                        raise ValueError('Unsupported uncompressed image encoding: '
                                         f'{fmt}')

                    msg = Image(header=header,
                                height=clip.height(),
                                width=clip.width(),
                                encoding=fmt,
                                is_bigendian=False,
                                step=step,
                                data=data)

                msg_serialized = typestore.serialize_cdr(msg, msgtype)

                writer.write(connection,
                             t_in_nanos,
                             msg_serialized)

