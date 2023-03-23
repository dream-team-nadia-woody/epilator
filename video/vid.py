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
from conversion import Conversions, Converter
from warnings import warn
__AGG_FUNCS = {
    'sum': lambda x: np.sum(x, axis=1),
    'mean': lambda x: np.mean(x, axis=1)

}


@dataclass
class Frame:
    frame: ArrayLike
    conversion: Converter
    frame_no: np.uint64
    seconds: np.float128

    @property
    def hue(self):
        return self.frame[:, :, 0]

    @property
    def lightness(self):
        return self.frame[:, :, 1]

    @property
    def saturation(self):
        return self.frame[:, :, 2]


class Video:
    '''A Class to store videos'''
    __vid: ArrayLike
    fps: float
    converter: Converter = Conversions.HLS.value

    def __init__(self, vid: Union[ArrayLike, Self],
                 fps: Union[float, None] = None,
                 converter:Conversions = Conversions.HLS) -> None:
        '''
        Creates a new video object
        ## Parameters:
        vid: either an `np.ndarray` or `Video` object
        fps: the frames per second of the video
        '''
        if isinstance(vid, Video):
            fps = vid.fps
            vid = vid.__vid
        if fps is None:
            raise ValueError(
                'FPS must be provided when converting from ArrayLike')
        self.__vid = vid
        self.fps = fps
        self.converter = converter.value

    @property
    def hue(self) ->Union[ArrayLike,None]:
        '''the hue values of the video, if applicable'''
        if (self.converter == Conversions.HLS.value
                or self.converter == Conversions.HSV.value):
            return self.__vid[:, :, :, 0]

    @property
    def saturation(self) -> Union[ArrayLike,None]:
        '''the saturation values of the video, if applicable'''
        if self.converter == Conversions.HLS.value:
            return self.__vid[:, :, :, 2]
        if self.converter == Conversions.HSV.value:
            return self.__vid[:,:,:,1]


    @property
    def lightness(self) -> Union[ArrayLike,None]:
        '''the lightness values of the video, if applicable'''
        if self.converter == Conversions.HLS.value
            return self.__vid[:, :, :, 1]
    @property
    def value(self) -> Union[ArrayLike,None]:
        '''the value values of the video, if applicable'''
        if self.converter == Conversions.HSV.value:
            return self.__vid[:, :, :, 2]

    @property
    def width(self)->int:
        '''the width of the video in pixels'''
        return self.__vid.shape[2]

    @property
    def height(self)->int:
        '''the height of the video in pixels'''
        return self.__vid.shape[1]

    @property
    def frame_count(self)->int:
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
    def from_file(cls, path: str, conversion: Conversions = Conversions.HLS) -> Self:
        '''
        Creates a new video object from file
        ## Parameters:
        path: string containing the path to the file
        conversion: `Conversions` enumeration value
        ## Returns:
        a new Video object
        '''
        vid, fps = VideoReader.get_vid(path, conversion)
        return cls(vid, fps)

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
        return Video(self.__vid[start:stop:step], self.fps)

    def __setitem__(self, frame_no: int, new_val: int):
        if new_val > 255:
            raise Exception(
                "255 is too great a value to be represented with np.uint8")
        self.__vid[frame_no] = new_val

    def mask(self, min_threshold: np.uint8 = 100,
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
        mask = cv.inRange(self.lightness, min_threshold,
                          max_threshold).reshape((-1, FRAME_Y, FRAME_X))
        arr = self.__vid.copy()
        arr[:, :, :, 1] = np.where(mask > 0, np.zeros_like(
            arr[:, :, :, 1]), arr[:, :, :, 1])
        return Video(arr, self.fps)

    def show(self, n_width: int = 5) -> Image:
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
            color_correct = cv.cvtColor(frame, cv.COLOR_HLS2RGB)
            img = Image.fromarray(color_correct)
            ret_img.paste(img, (x, y, x + self.width, y + self.height))
        return ret_img

    def pct_change(self, n: int):
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

        agg_arr = self.agg_lightness()
        shifted_arr = np.roll(agg_arr, n)
        shifted_arr[:n] = np.nan
        percent_change = (agg_arr - shifted_arr) / shifted_arr

        return percent_change

    def agg_lightness(self, agg: Union[Callable, str] = __AGG_FUNCS['mean'],
                      **kwargs)->ArrayLike:
        '''
        Aggregates the lightness channel by a given function
        ## Parameters:
        agg: either a string representing the function
        to use in __AGG_FUNCS or a Callable function
        which accepts an `ndarray` and returns an `ndarray`.
        ## Returns:
        the aggregated values of the video
        '''
        inline = self.lightness.reshape((self.frame_count, -1))
        if isinstance(agg, Callable):
            return agg(inline, **kwargs)
        return __AGG_FUNCS[agg](inline, **kwargs)