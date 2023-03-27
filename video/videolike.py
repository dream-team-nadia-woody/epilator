from typing import Self, Tuple, Union
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike

from video.channel import Channel
from video.conversion import Conversions, Converter


class VideoLike:
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
        self._vid = vid
        self.fps = fps
        if isinstance(converter, Conversions):
            self.converter = converter.value
        else:
            self.converter = converter

    def __slices_except_last(self, n: int) -> Tuple[slice]:
        slices = [slice(None)] * self._vid.ndim
        slices[len(slices)-1] = slice(n)
        return tuple(slices)

    @property
    def hue(self) -> Channel:
        return Channel(self._vid[self.__slices_except_last(0)],
                       self.converter)

    @property
    def saturation(self) -> Channel:
        if self.converter == Conversions.HLS.value:
            return Channel(self._vid[self.__slices_except_last(2)],
                           self.converter)
        elif self.converter == Conversions.HSV.value:
            return Channel(self._vid[self.__slices_except_last(1)],
                           self.converter)
        else:
            raise ValueError(
                'This colorspace does not have a saturation channel')

    @property
    def lightness(self) -> Channel:
        if self.converter == Conversions.HLS.value:
            return Channel(self._vid[self.__slices_except_last(1)],
                           self.converter)
        else:
            raise ValueError(
                'This colorspace does not have a lightness channel')

    @property
    def value(self) -> Channel:
        if self.converter == Conversions.HSV.value:
            return Channel(self._vid[self.__slices_except_last(2)],
                           self.converter)
        else:
            raise ValueError('This colorspace does not have a value channel')

    @property
    def red(self) -> Channel:
        if self.converter == Conversions.RGB.value:
            return Channel(self._vid[self.__slices_except_last(0)],
                           self.converter)
        elif self.converter == Conversions.BGR.value:
            return Channel(self._vid[self.__slices_except_last(2)],
                           self.converter)
        else:
            raise ValueError('This colorspace does not have a red channel')

    @property
    def green(self) -> Channel:
        if (self.converter == Conversions.RGB.value
                or self.converter == Conversions.BGR.value):
            return Channel(self._vid[self.__slices_except_last(1)],
                           self.converter)
        else:
            raise ValueError('This colorspace does not have a green channel')

    @property
    def blue(self) -> Channel:
        if self.converter == Conversions.RGB.value:
            return Channel(self._vid[self.__slices_except_last(2)],
                           self.converter)
        elif self.converter == Conversions.BGR.value:
            return Channel(self._vid[self.__slices_except_last(0)],
                           self.converter)
        else:
            raise ValueError('This colorspace does not have a blue channel')

    @property
    def width(self) -> int:
        '''the width of the video in pixels'''
        width_index = self._vid.ndim - 2
        return self._vid.shape[width_index]

    @property
    def height(self) -> int:
        '''the height of the video in pixels'''
        height_index = self._vid.ndim - 3
        return self._vid.shape[height_index]

    @property
    def arr(self) -> ArrayLike:
        '''the underlying numpy array'''
        warn('using this property is discouraged. '
             'Are you sure you want to use it?')
        return self._vid

    @property
    def shape(self) -> Tuple[int]:
        '''the shape of the video'''
        return self._vid.shape

    def copy(self) -> Self:
        '''returns a copy of the video'''
        return type(self)(self._vid.copy(), self.fps, self.converter)
