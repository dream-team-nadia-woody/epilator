from typing import Callable, List, Self, Tuple, Union
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from video.channel import Channel, AGG_FUNCS, AggregatorFunc
from video.conversion import Conversions, Converter
from video.reader import VideoReader, FRAME_X,FRAME_Y
import cv2 as cv
from PIL import Image


class VideoLike(ABC):
    '''An abstract class representing a video
    Video and Frame are instances of this class'''
    vid: ArrayLike
    fps: float
    converter: Converter

    def __init__(self, vid: Union[ArrayLike, Self],
                 fps: Union[float, None] = None,
                 converter: Union[Converter,
                                  Conversions] = Conversions.HLS,
                 segments: int = 1) -> None:
        if type(self) == VideoLike:
            raise TypeError('VideoLike is an abstract class')
        if isinstance(vid, VideoLike):
            fps = vid.fps
            converter = vid.converter
            vid = vid.vid
        self.vid = vid
        self.fps = fps
        if isinstance(converter, Conversions):
            self.converter = converter.value
        else:
            self.converter = converter
        self.segments = segments

    def __channel_at_index(self, n: int) -> Tuple[slice]:
        slices = [slice(None)] * self.vid.ndim
        slices[len(slices)-1] = slice(n, n+1, 1)
        return tuple(slices)

    @property
    def hue(self) -> Channel:
        '''Returns the hue channel of the video'''
        if (self.converter == Conversions.HLS.value
                or self.converter == Conversions.HSV.value):
            return Channel(self.converter.channel_names[0],
                           self.vid[self.__channel_at_index(0)],
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
                       self.vid[self.__channel_at_index(index)],
                       self.converter)

    @property
    def lightness(self) -> Channel:
        if self.converter == Conversions.HLS.value:
            return Channel(self.converter.channel_names[1],
                           self.vid[self.__channel_at_index(1)],
                           self.converter)
        else:
            raise ValueError(
                'This colorspace does not have a lightness channel')

    @property
    def value(self) -> Channel:
        if self.converter == Conversions.HSV.value:
            return Channel(self.converter.channel_names[2],
                           self.vid[self.__channel_at_index(2)],
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
                       self.vid[self.__channel_at_index(index)],
                       self.converter)

    @property
    def green(self) -> Channel:
        if (self.converter == Conversions.RGB.value
                or self.converter == Conversions.BGR.value):
            return Channel(self.converter.channel_names[1],
                           self.vid[self.__channel_at_index(1)],
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
                       self.vid[self.__channel_at_index(index)],
                       self.converter)

    @property
    def gray(self) -> Channel:
        if self.converter != Conversions.GRAY.value:
            gray = self.grayscale()
        return Channel('gray', self.vid[..., 0], self.converter)

    @property
    def width(self) -> int:
        '''the width of the video in pixels'''
        width_index = self.vid.ndim - 2
        return self.vid.shape[width_index]

    @property
    def height(self) -> int:
        '''the height of the video in pixels'''
        height_index = self.vid.ndim - 3
        return self.vid.shape[height_index]

    @property
    def shape(self) -> Tuple[int]:
        '''the shape of the video'''
        return self.vid.shape

    def copy(self) -> Self:
        '''returns a copy of the video'''
        return type(self)(self.vid.copy(), self.fps, self.converter)

    @classmethod
    def from_file(cls, path: str,
                  converter: Conversions = Conversions.HSV,
                  resize:Tuple[int,int] = (FRAME_X,FRAME_Y)) -> Self:
        '''
        Creates a new video object from file
        ## Parameters:
        path: string containing the path to the file
        conversion: `Conversions` enumeration value
        ## Returns:
        a new Video object
        '''
        vid, fps = VideoReader.get_vid(path, converter.value.load,resize)
        return cls(vid, fps, converter)

    def get_channel(self, channel_name: Union[str, int, Channel]) -> Channel:
        '''returns the channel with the given name'''
        match channel_name:
            case Channel():
                return channel_name
            case int():
                return Channel(
                    self.converter.channel_names[channel_name],
                    self.vid[self.__channel_at_index(channel_name)],
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
            channel = ret_vid.get_channel(channel)
            mask = channel.get_mask(min_threshold, max_threshold)
            mask = mask.reshape(*mask.shape[0:mask.ndim - 1])
            if channel.channel_name in ['value', 'lightness']:
                index = self.converter.channel_names.index(
                    channel.channel_name)
                
                ret_vid.vid[...,index][~mask] = 0
                return ret_vid
            for i in range(3):
                chan = ret_vid.vid[..., i]
                chan[~mask] = 0
            return ret_vid
        for i in range(3):
            mask = ret_vid.get_channel(i).mask(
                min_threshold, max_threshold).channel
            ret_vid.vid[..., i] = mask[..., 0]
            return ret_vid

    def agg(self, func: Union[AggregatorFunc, None] = AGG_FUNCS['sum'],
            channel: Union[str, int, None] = None) -> Self:
        if isinstance(func, str):
            func = AGG_FUNCS[func]
        if channel is not None:
            channel = self.get_channel(channel)
            return channel.agg(func)
        if self.vid.ndim < 4:
            ret_arr = np.zeros((1, 3), dtype=np.uint64)
        else:
            ret_arr = np.zeros((self.vid.shape[0], 3), dtype=np.uint64)
        for i in range(3):
            channel = self.get_channel(i)
            ret_arr[:, i] = channel.agg(func)
        return ret_arr

    def pct_change(self, periods: int, channel: Union[str, int, None] = None,
                   agg: AggregatorFunc = AGG_FUNCS['mean']) -> Self:
        if channel is not None:
            channel = self.get_channel(channel)
            return channel.pct_change(periods, agg)

        if self.vid.ndim < 4:
            ret_arr = np.zeros((1, 3), dtype=np.float64)
        else:
            ret_arr = np.zeros((self.vid.shape[0], 3), dtype=np.float64)

        for i in range(3):
            channel = self.get_channel(i)
            ret_arr[:, i] = channel.pct_change(periods, agg)

        return ret_arr
    
    def _get_img(self, frame: ArrayLike) -> Image:
        if self.converter.display > 0:
            frame = cv.cvtColor(frame, self.converter.display)
        return Image.fromarray(frame)
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

    def reconvert(self, new_conversion: Union[
            Conversions,
            Converter]) -> Self:
        ret_vid = self.copy()
        if isinstance(new_conversion, Conversions):
            new_conversion = new_conversion.value
        if ret_vid.converter.bgr > 0:
            for index, frame in enumerate(ret_vid.vid):
                ret_vid.vid[index] = cv.cvtColor(frame,
                                                  ret_vid.converter.bgr)
        if new_conversion.load > 0:
            for index, frame in enumerate(ret_vid.vid):
                ret_vid.vid[index] = cv.cvtColor(frame,
                                                  ret_vid.converter.load)
        ret_vid.converter = new_conversion
        return ret_vid

    def grayscale(self) -> Self:
        ret_vid = np.zeros(self.vid.shape[:-1], dtype=np.uint8)
        for index, frame in enumerate(self.vid):
            ret_vid[index] = cv.cvtColor(frame,  cv.COLOR_BGR2GRAY)
        return type(self)(ret_vid, self.fps, Conversions.GRAY)

    @abstractmethod
    def segment(self, n: int) -> List[Self]:
        '''splits the video into n equal parts'''
        pass

    def blur(self,kernel_x:int, kernel_y:Union[int, None ] = None)->Self:
        '''returns a copy of itself with the video blurred '''
        ret_vid = self.copy()
        if kernel_y is None:
            kernel_y = kernel_x
        for index, frame in enumerate(ret_vid.vid):
            ret_vid.vid[index] = cv.blur(frame,(kernel_x,kernel_y))
        return ret_vid