from typing import Callable, Self, Tuple, Union
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from video.channel import Channel, AGG_FUNCS, AggregatorFunc
from video.conversion import Conversions, Converter
from video.reader import VideoReader
import cv2 as cv
from PIL import Image


class VideoLike(ABC):
    '''An abstract class representing a video
    Video and Frame are instances of this class'''
    _vid: ArrayLike
    fps: float
    converter: Converter

    def __init__(self, vid: Union[ArrayLike, Self],
                 fps: Union[float, None] = None,
                 converter: Union[Converter,
                                  Conversions] = Conversions.HLS) -> None:
        if type(self) == VideoLike:
            raise TypeError('VideoLike is an abstract class')
        if isinstance(vid, VideoLike):
            fps = vid.fps
            converter = vid.converter
            vid = vid.__vid
        self.__vid = vid
        self.fps = fps
        if isinstance(converter, Conversions):
            self.converter = converter.value
        else:
            self.converter = converter

    def __channel_at_index(self, n: int) -> Tuple[slice]:
        slices = [slice(None)] * self.__vid.ndim
        slices[len(slices)-1] = slice(n, n+1, 1)
        return tuple(slices)

    @property
    def hue(self) -> Channel:
        '''Returns the hue channel of the video'''
        if (self.converter == Conversions.HLS.value
                or self.converter == Conversions.HSV.value):
            return Channel(self.converter.channel_names[0],
                           self.__vid[self.__channel_at_index(0)],
                           self.converter)
        else:
            raise ValueError('This colorspace does not have a hue channel')

    @property
    def saturation(self) -> Channel:
        if self.converter == Conversions.HLS.value:
            index = 2
        elif self.converter == Conversions.HSV.value:
            index = 1
        else:
            raise ValueError(
                'This colorspace does not have a saturation channel')
        return Channel(self.converter.channel_names[index],
                       self.__vid[self.__channel_at_index(index)],
                       self.converter)

    @property
    def lightness(self) -> Channel:
        if self.converter == Conversions.HLS.value:
            return Channel(self.converter.channel_names[1],
                           self.__vid[self.__channel_at_index(1)],
                           self.converter)
        else:
            raise ValueError(
                'This colorspace does not have a lightness channel')

    @property
    def value(self) -> Channel:
        if self.converter == Conversions.HSV.value:
            return Channel(self.converter.channel_names[2],
                           self._vid[self.__channel_at_index(2)],
                           self.converter)
        else:
            raise ValueError('This colorspace does not have a value channel')

    @property
    def red(self) -> Channel:
        if self.converter == Conversions.RGB.value:
            index = 0
        elif self.converter == Conversions.BGR.value:
            index = 2
        else:
            raise ValueError('This colorspace does not have a red channel')
        return Channel(self.converter.channel_names[index],
                       self.__vid[self.__channel_at_index(index)],
                       self.converter)

    @property
    def green(self) -> Channel:
        if (self.converter == Conversions.RGB.value
                or self.converter == Conversions.BGR.value):
            return Channel(self.converter.channel_names[1],
                           self.__vid[self.__channel_at_index(1)],
                           self.converter)
        else:
            raise ValueError('This colorspace does not have a green channel')

    @property
    def blue(self) -> Channel:
        if self.converter == Conversions.RGB.value:
            index = 2
        elif self.converter == Conversions.BGR.value:
            index = 0
        else:
            raise ValueError('This colorspace does not have a blue channel')
        return Channel(self.converter.channel_names[index],
                       self.__vid[self.__channel_at_index(index)],
                       self.converter)

    @property
    def width(self) -> int:
        '''the width of the video in pixels'''
        width_index = self.__vid.ndim - 2
        return self.__vid.shape[width_index]

    @property
    def height(self) -> int:
        '''the height of the video in pixels'''
        height_index = self.__vid.ndim - 3
        return self.__vid.shape[height_index]

    @property
    def _vid(self) -> ArrayLike:
        '''the numpy array representing the video'''
        return self.__vid

    @property
    def shape(self) -> Tuple[int]:
        '''the shape of the video'''
        return self.__vid.shape

    def copy(self) -> Self:
        '''returns a copy of the video'''
        return type(self)(self._vid.copy(), self.fps, self.converter)

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

    def get_channel(self, channel_name: Union[str, int, Channel]) -> Channel:
        '''returns the channel with the given name'''
        match channel_name:
            case Channel():
                return channel_name
            case int():
                return Channel(
                    self.converter.channel_names[channel_name],
                    self.__vid[self.__channel_at_index(channel_name)],
                    self.converter)
            case 'hue' | 'h':
                return self.hue
            case 'saturation' | 'sat' | 's':
                return self.saturation
            case 'lightness' | 'light' | 'l':
                return self.lightness
            case 'value' | 'val' | 'v':
                return self.value
            case 'red' | 'r':
                return self.red
            case 'green' | 'g':
                return self.green
            case 'blue' | 'b':
                return self.blue
            case _:
                raise ValueError('Invalid channel name')

    def mask(self, channel: Union[str, int, None] = None,
             min_threshold: int = 190, max_threshold: int = 255) -> Self:
        ret_vid = self.copy()
        if channel is not None:
            masked_channel = ret_vid.get_channel(
                channel).mask(min_threshold, max_threshold)
            ret_vid.__vid[..., self.converter.channel_names.index(
                channel)] = masked_channel.channel[..., 0]
            return ret_vid
        for i in range(3):
            masked_channel = ret_vid.get_channel(i).mask(
                min_threshold, max_threshold).channel
            ret_vid.__vid[..., i] = masked_channel[..., 0]
            return ret_vid

    def agg(self, func: Union[AggregatorFunc, None] = AGG_FUNCS['sum'],
            channel: Union[str, int, None] = None) -> Self:
        if isinstance(func, str):
            func = AGG_FUNCS[func]
        if channel is not None:
            channel = self.get_channel(channel)
            return channel.agg(func)
        if self.__vid.ndim < 4:
            ret_arr = np.zeros((1, 3), dtype=np.uint64)
        else:
            ret_arr = np.zeros((self.__vid.shape[0], 3), dtype=np.uint64)
        for i in range(3):
            channel = self.get_channel(i)
            ret_arr[:, i] = channel.agg(func)
        return ret_arr


    def pct_change(self, periods: int, channel: Union[str, int, None] = None,
                agg: AggregatorFunc = AGG_FUNCS['sum']) -> Self:
        if channel is not None:
            channel = self.get_channel(channel)
            return channel.pct_change(periods, agg)

        if self._vid.ndim < 4:
            ret_arr = np.zeros((1, 3), dtype=np.float64)
        else:
            ret_arr = np.zeros((self._vid.shape[0], 3), dtype=np.float64)

        for i in range(3):
            channel = self.get_channel(i)
            ret_arr[:, i] = channel.pct_change(periods, agg)

        return ret_arr

    @abstractmethod
    def show(self, n_width: int = 5) -> Image:
        '''shows the video'''
        pass

    def difference(self, n: int, channel: Union[ArrayLike, str, int],
                   agg: AggregatorFunc = AGG_FUNCS['mean']):
        '''
        Returns the difference in lightness between each `n` frames.

        ## Parameters:
        n: integer representing the number of frames to look ahead
        ## Returns:
        an `ndarray` with the percentage change in frames.
        '''
        if n < 1:
            raise ValueError("n must be greater than or equal to 1")
        channel = self.get_channel(channel)
        return channel.difference(n)
