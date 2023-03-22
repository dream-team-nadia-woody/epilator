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

AGG_FUNCS = {
    'sum': lambda x: np.sum(x, axis=1),
    'mean': lambda x: np.mean(x, axis=1)

}


@dataclass
class Frame:
    frame: ArrayLike
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
    __vid: ArrayLike
    fps: float
    convert_in: int = cv.COLOR_BGR2HLS

    def __init__(self, vid: Union[ArrayLike, Self],
                 fps: Union[float, None] = None) -> None:
        if isinstance(vid, Video):
            fps = vid.fps
            vid = vid.__vid
        if fps is None:
            raise ValueError(
                'FPS must be provided when converting from ArrayLike')
        self.__vid = vid
        self.fps = fps

    @property
    def hue(self) -> ArrayLike:
        return self.__vid[:, :, :, 0]

    @property
    def saturation(self) -> ArrayLike:
        return self.__vid[:, :, :, 2]

    @property
    def lightness(self) -> ArrayLike:
        return self.__vid[:, :, :, 1]

    @property
    def width(self):
        return self.__vid.shape[2]

    @property
    def height(self):
        return self.__vid.shape[1]

    @property
    def frame_count(self):
        return self.__vid.shape[0]

    @property
    def arr(self):
        return self.__vid

    @property
    def shape(self):
        return self.__vid.shape

    @classmethod
    def from_file(cls, path: str, conversion: int = cv.COLOR_BGR2HLS) -> Self:
        vid, fps = VideoReader.get_vid(path, conversion)
        return cls(vid, fps)

    def __getitem__(self, frame_no: Union[int, slice]) -> Union[Frame, Self]:
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
        if n < 1:
            raise ValueError("n must be greater than or equal to 1")

        agg_arr = self.agg_lightness()
        shifted_arr = np.roll(agg_arr, n)
        shifted_arr[:n] = np.nan
        percent_change = (agg_arr - shifted_arr) / shifted_arr

        return percent_change

    def agg_lightness(self, agg: Union[Callable, str] = AGG_FUNCS['mean'],
                      **kwargs):
        inline = self.lightness.reshape((self.frame_count, -1))
        if isinstance(agg, Callable):
            return agg(inline, **kwargs)
        return AGG_FUNCS[agg](inline, **kwargs)


def get_video_from_iterator(path: str) -> Tuple[ArrayLike, float]:
    vid = VideoReader(path)
    fps = vid.video.get(cv.CAP_PROP_FPS)
    video = np.fromiter(vid, np.ndarray)
    return np.stack(video), fps
