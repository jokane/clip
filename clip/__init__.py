"""
This is a library for manipulating and generating short video clips. It can
read video in any format supported by ffmpeg, and provides abstractions for
things like cropping, superimposing, adding text, trimming, fading in and out,
fading between clips, etc.  Additional effects can be achieved by filtering the
frames through custom functions.
"""

from .alpha import *
from .audio import *
from .base import *
from .chain import *
from .color import *
from .crop_slice import *
from .dimensions import *
from .fade import *
from .ffmpeg import *
from .filter import *
from .from_file import *
from .glob import *
from .hold import *
from .ken_burns import *
from .loop import *
from .pdf import *
from .resample import *
from .rosbag import *
from .save_audio import *
from .save_mp4 import *
from .save_gif import *
from .scale import *
from .spin import *
from .stack import *
from .subtitles import *
from .superimpose import *
from .text import *
from .util import *
from .video import *
from .zip_clip import *
