from dataclasses import dataclass
from enum import Enum
from typing import Callable, Self, Tuple, Union
from warnings import warn

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from PIL import Image
from typing_extensions import override

from video.channel import AGG_FUNCS, AggregatorFunc, Channel
from video.conversion import Conversions, Converter
from video.frame import Frame
from video.generator import vid_frame
from video.reader import FRAME_X, FRAME_Y, VideoReader
from video.videolike import VideoLike


class Video(VideoLike):
    '''A Class to store videos'''
    fps: float
    converter: Converter
    start_time: np.float128
    end_time: np.float128

    def __init__(self, vid: Union[ArrayLike, Self],
                 fps: Union[float, None] = None,
                 converter: Conversions = Conversions.HLS,
                 start_time: Union[float, None] = None,
                 end_time: Union[float, None] = None,
                 ) -> None:
        '''
        Creates a new video object
        ## Parameters:
        vid: either an `np.ndarray` or `Video` object
        fps: the frames per second of the video
        '''
        super().__init__(vid, fps, converter)
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = vid.shape[0] / fps
        self.start_time = start_time
        self.end_time = end_time

    @property
    def frame_count(self) -> int:
        '''the number of frames in the video'''
        return self.shape[0]

    @classmethod
    def from_file(cls, path: str,
                  converter: Conversions = Conversions.HSV) -> Self:
        '''
        Creates a new video object from file
        ## Parameters:
        path: string containing the path to the file
        conversion: `Conversions` enumeration value
        ## Returns:
        a new Video object
        '''
        vid, fps = VideoReader.get_vid(path, converter.value.load)
        return cls(vid, fps, converter)

    def __getitem__(self, frame_no: Union[int, slice]) -> Union[Frame, Self]:
        '''
        Allows for bracket notation in accessing `Video`
        ## Parameters:
        frame_no: either an integer or slice of the values to access
        ## Returns:
        a `Frame` object if accessing a single frame of the video,
        else a `Video` referencing the slice of the video
        '''
        if isinstance(frame_no, int):
            frame = self._vid[frame_no]
            seconds = np.float128(frame_no / self.fps)
            return Frame(frame, self.fps, self.converter, frame_no)
        start, stop, step = frame_no.indices(self._vid.shape[0])
        return Video(self._vid[start:stop:step], self.fps, self.converter)

    def __setitem__(self, frame_no: int, new_val: int):
        if new_val > 255:
            raise Exception(
                "255 is too great a value to be represented with np.uint8")
        self._vid[frame_no] = np.full(
            (self.height, self.width, 3), new_val, dtype=np.uint8)

    def copy(self) -> Self:
        return Video(self._vid.copy(), self.fps, self.converter)

    def mask_channel(self, channel: Union[str, int],
                     min_threshold: int = 190,
                     max_threshold: int = 255) -> 'Video':
        if isinstance(channel, str):
            channel_idx = self.get_channel_index(channel)
        elif isinstance(channel, int):
            channel_idx = channel
        else:
            raise ValueError('Channel value is unsupported')

        masked_vid = self.copy()
        for i in range(masked_vid._vid.shape[0]):
            frame = masked_vid._vid[i]
            channel_data = frame[:, :, channel_idx]
            lower_bound = np.full(channel_data.shape,
                                  min_threshold, dtype=np.uint8)
            upper_bound = np.full(channel_data.shape,
                                  max_threshold, dtype=np.uint8)
            mask = cv.inRange(channel_data, lower_bound, upper_bound)
            frame[:, :, channel_idx] = np.where(mask > 0, channel_data, 0)

        return masked_vid

    def show(self, n_width: int = 5) -> Image:
        '''
        Shows a sequence of frames from a `Video`,
        `np.ndarray` or `pd.DataFrame`
        ## Parameters:
        arr: Either a `Video`, `np.ndarray` or `pd.DataFrame`
        containing the video to be displayed
        n: the number of images to display horizontally
        ## Returns:
        a `PIL.Image` object of dimensions :

        `(n * [frame width], [frame height] * [frame count] // n)`'''
        if self.frame_count < n_width:
            ret_width = self.width * self.frame_count
            ret_height = self.height
        else:
            ret_width = self.width * n_width
            ret_height = self.frame_count * self.height // n_width
        ret_img = Image.new('RGB', (ret_width, ret_height))
        for index, frame in enumerate(self._vid):
            x = self.width * (index % n_width)
            y = frame.shape[0] * (index // n_width)
            color_correct = cv.cvtColor(frame, self.converter.display)
            img = Image.fromarray(color_correct)
            ret_img.paste(img, (x, y, x + self.width, y + self.height))
        return ret_img

    def get_channel(self, channel: Union[ArrayLike, str, int]):
        match channel:
            case int():
                channel = Channel(self._vid[:, :, :, channel],
                                  self.converter)
            case np.ndarray():
                channel = Channel(channel, self.converter)
            case Channel():
                pass
            case 'hue' | 'h':
                channel = self.hue
            case 'saturation' | 'sat' | 's':
                channel = self.saturation
            case 'value' | 'v':
                channel = self.value
            case 'lightness' | 'l':
                channel = self.lightness
            case _:
                raise ValueError('Channel Value is unsupported')
        return channel

    def get_channel_index(self, channel_name: str) -> int:
        match channel_name.lower():
            case 'hue' | 'h':
                return 0
            case 'saturation' | 'sat' | 's':
                if self.converter == Conversions.HLS.value:
                    return 2
                elif self.converter == Conversions.HSV.value:
                    return 1
                else:
                    raise ValueError('Unsupported channel')
            case 'value' | 'v':
                return 2
            case 'lightness' | 'l':
                return 1
            case _:
                raise ValueError('unsupported channel')

    def agg(self, channel: Union[ArrayLike, str, int],
            agg: Union[Callable, str]) -> ArrayLike:
        '''
        Aggregates the given the `channel` by `agg` function
        ## Parameters:
        channel: either a string or integer representing the channel
        to be aggregated.
        agg: function with which to aggregate each frame's values.
        ## Returns:
        an `ndarray` containing the aggregated results of `channel`
        '''
        channel = self.get_channel(channel)
        return channel.agg(agg)

    def pct_change(self, n: int, channel: Union[ArrayLike, str, int],
                   agg: AggregatorFunc = AGG_FUNCS['mean']):
        '''
        Returns the percentage change in lightness between each `n` frames.
        Comperable to `pd.DataFrame.pct_change(n)`
        ## Parameters:
        n: integer representing the number of frames to look ahead
        ## Returns:
        an `ndarray` with the percentage change in frames.
        '''
        if n < 1:
            raise ValueError("n must be greater than or equal to 1")
        channel = self.get_channel(channel)
        return channel.pct_change(n)
