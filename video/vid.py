from typing import Callable, Self, Tuple, Union
from typing_extensions import override
import cv2 as cv
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from dataclasses import dataclass
from PIL import Image
from enum import Enum
from video.reader import VideoReader, FRAME_X, FRAME_Y
from video.conversion import Conversions, Converter
from warnings import warn
from video.generator import vid_frame
AGG_FUNCS = {
    'sum': lambda x: np.sum(x, axis=1),
    'mean': lambda x: np.mean(x, axis=1)

}

AggregatorFunc = Union[Callable, str]


@dataclass
class Channel:
    '''a wrapper class representing
    a color channel of a `Video`
    '''
    channel: np.ndarray
    conversion: Converter

    def agg(self, agg: AggregatorFunc) -> ArrayLike:
        inline = self.channel.reshape((self.channel.shape[0], -1))
        if isinstance(agg, Callable):
            return agg(inline)
        return AGG_FUNCS[agg](inline)

    def pct_change(self, n: int,
                   agg: AggregatorFunc = AGG_FUNCS['mean']) -> ArrayLike:
        agg_arr = self.agg(agg)
        shifted_arr = np.roll(agg_arr, n)
        shifted_arr[:n] = np.nan
        return (agg_arr - shifted_arr) / shifted_arr


@dataclass
class Frame:
    '''
    A wrapper class representing the single frame of
    a video. This prevents issues with reshaping
    of the array.
    '''
    frame: ArrayLike
    conversion: Converter
    frame_no: np.uint64
    seconds: np.float128

    @property
    def hue(self):
        return Channel(self.frame[:, :, 0], self.conversion)

    @property
    def lightness(self):
        return Channel(self.frame[:, :, 1], self.conversion)

    @property
    def saturation(self):
        return Channel(self.frame[:, :, 2], self.conversion)


class Video:
    '''A Class to store videos'''
    __vid: ArrayLike
    fps: float
    converter: Converter

    def __init__(self, vid: Union[ArrayLike, Self],
                 fps: Union[float, None] = None,
                 converter: Conversions = Conversions.HLS) -> None:
        '''
        Creates a new video object
        ## Parameters:
        vid: either an `np.ndarray` or `Video` object
        fps: the frames per second of the video
        '''
        if fps is None:
            raise ValueError(
                'FPS must be provided when converting from ArrayLike')
        self.__vid = vid
        self.fps = fps
        if isinstance(converter, Conversions):
            converter = converter.value
        self.converter = converter

    @property
    def hue(self) -> Union[Channel, None]:
        '''the hue values of the video, if applicable'''
        if (self.converter == Conversions.HLS.value
                or self.converter == Conversions.HSV.value):
            return Channel(self.__vid[:, :, :, 0], self.converter)

    @property
    def saturation(self) -> Union[Channel, None]:
        '''the saturation values of the video, if applicable'''
        if self.converter == Conversions.HLS.value:
            return Channel(self.__vid[:, :, :, 2], self.converter)
        if self.converter == Conversions.HSV.value:
            return Channel(self.__vid[:, :, :, 1], self.converter)

    @property
    def lightness(self) -> Union[Channel, None]:
        '''the lightness values of the video, if applicable'''
        if self.converter == Conversions.HLS.value:
            return Channel(self.__vid[:, :, :, 1], self.converter)

    @property
    def value(self) -> Union[Channel, None]:
        '''the value values of the video, if applicable'''
        if self.converter == Conversions.HSV.value:
            return Channel(self.__vid[:, :, :, 2], self.converter)

    @property
    def width(self) -> int:
        '''the width of the video in pixels'''
        return self.__vid.shape[2]

    @property
    def height(self) -> int:
        '''the height of the video in pixels'''
        return self.__vid.shape[1]

    @property
    def frame_count(self) -> int:
        '''the number of frames in the video'''
        return self.__vid.shape[0]

    @property
    def arr(self):
        '''the video array NOT ADVISED'''
        warn('used for debugging, not recommended in production')
        return self.__vid

    @property
    def shape(self):
        '''the shape of the underlying array'''
        return self.__vid.shape

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
            frame = self.__vid[frame_no]
            seconds = np.float128(frame_no / self.fps)
            return Frame(frame, frame_no, seconds)
        start, stop, step = frame_no.indices(self.__vid.shape[0])
        return Video(self.__vid[start:stop:step], self.fps, self.converter)

    def __setitem__(self, frame_no: int, new_val: int):
        if new_val > 255:
            raise Exception(
                "255 is too great a value to be represented with np.uint8")
        self.__vid[frame_no] = new_val

    def copy(self) -> Self:
        return Video(self.__vid.copy(), self.fps, self.converter)

    def mask(self, channel: Union[ArrayLike, Channel, int, str],
             min_threshold: np.uint8 = 190,
             max_threshold: np.uint8 = 255) -> Self:
        '''
        Returns a masked version of the video
        ## Parameters:
        min_threshold *(optional)*: the minimum `lightness` value to
        include in masked video
        max_threshold *(optional)*: the maximum `lightness` value to
        include in masked video
        ## Returns:
        a new `Video` object containing the masked data
        '''
        min = np.zeros(3)
        max = np.zeros(3)
        min[channel] = min_threshold
        max[channel] = max_threshold
        mask = cv.inRange(self.__vid, min, max)
        mask = np.where(mask[:,:,:,channel] > 0, self.__vid, mask)
        return Video(mask, self.fps, self.converter)


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
        for index, frame in enumerate(self.__vid):
            x = self.width * (index % n_width)
            y = frame.shape[0] * (index // n_width)
            color_correct = cv.cvtColor(frame, self.converter.display)
            img = Image.fromarray(color_correct)
            ret_img.paste(img, (x, y, x + self.width, y + self.height))
        return ret_img

    def get_channel(self, channel: Union[ArrayLike, str, int]):
        match channel:
            case int():
                channel = Channel(self.__vid[:, :, :, channel],
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
