from typing import List, Self, Tuple, Union
from typing_extensions import override
import cv2 as cv
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from dataclasses import dataclass
from collections import deque
from PIL import Image
FRAME_X = 100
FRAME_Y = 100


class VideoReader:
    video: cv.VideoCapture

    def __init__(self, path: str):
        self.video = cv.VideoCapture(path)

    def __del__(self):
        self.video.release()

    def __iter__(self):
        if not self.video.isOpened():
            raise StopIteration
        return self

    def __next__(self):
        end, frame = self.video.read()
        if not end:
            raise StopIteration
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        frame = cv.resize(frame, (FRAME_X, FRAME_Y),
                          interpolation=cv.INTER_NEAREST)
        return frame

    @classmethod
    def get_vid(cls, path: str,
                conversion: int = cv.COLOR_BGR2HLS) -> ArrayLike:
        vid_reader = cls(path)
        if not vid_reader.video.isOpened():
            raise EOFError('File not opened')
        frames = []
        fps = vid_reader.video.get(cv.CAP_PROP_FPS)
        while True:
            flag, frame = vid_reader.video.read()
            if not flag:
                break
            if conversion > 0:
                frame = cv.cvtColor(frame, conversion)
            frame = cv.resize(frame, (FRAME_X, FRAME_Y),
                              interpolation=cv.INTER_NEAREST)
            frames.append(frame)
        return np.stack(np.asarray(frames, dtype=np.uint8)), int(round(fps))


def get_video_from_iterator(path: str) -> Tuple[ArrayLike, float]:
    vid = VideoReader(path)
    fps = vid.video.get(cv.CAP_PROP_FPS)
    video = np.fromiter(vid, np.ndarray)
    return np.stack(video), fps


def get_vid_df(vid: Union[str, ArrayLike], fps: int = 30,
               conversion: int = cv.COLOR_BGR2HLS,
               rename: List[str] = [
        'hue', 'lightness', 'saturation']) -> pd.DataFrame:
    if isinstance(vid, str):
        vid, fps = VideoReader(vid).get_vid(conversion)
    frames = vid.shape[0]
    height = vid.shape[1]
    width = vid.shape[2]
    df = pd.DataFrame(vid.reshape((-1, 3)))
    df['frame'] = df.index // (width * height)
    df['x'] = df.index % width
    df['y'] = df.index // width % height
    rename = {key: value for key, value in enumerate(rename)}
    df = df.set_index(['frame', 'y', 'x']).rename(columns=rename)
    df.attrs['height'] = height
    df.attrs['width'] = width
    df.attrs['fps'] = fps
    df.attrs['conversion'] = conversion
    return df


def get_mask(img: np.array):
    '''
    get the lightness mask from hls image
    '''

    Lchannel = img[:, :, 1]
    mask = cv.inRange(Lchannel, 160, 255)
    # mask = np.where(255, 1, 0)

    return mask


def add_mask(df: pd.DataFrame) -> pd.DataFrame:
    '''
    calls get_frame() to generate mask for each frame. s
    aves results in np.array of 1 and 0
    where 1 - light pixel, 0 - dark pixel
    '''
    # let's try the same but through numpy array
    narr = np.empty((0,), dtype=np.uint8)
    w = df.attrs['width']
    h = df.attrs['height']
    # loop through the frames
    for f in df.index.levels[0]:
        # save an image to the variable 'frame'
        frame = df.loc[f].iloc[:, :].to_numpy().reshape(w, h, 3)
        # get the mask
        mask = get_mask(frame)
        narr = np.concatenate([narr, mask.reshape(-1)])
    return df.assign(masked_values=narr)


def add_seconds(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds columns 'seconds' that shows the second of the video
    '''
    return df.assign(
        seconds=lambda x: x.index.get_level_values('frame') // x.attrs['fps']
    )


def get_exploration_df(vid: Union[str, ArrayLike], fps: int = 30,
                       conversion: int = cv.COLOR_BGR2HLS) -> pd.DataFrame:
    '''
    returns a data frame of the video with mask values and seconds added
    '''
    df = get_vid_df(vid)
    df = add_mask(df)
    df = add_seconds(df)
    df.masked_values.replace({255: 1}, inplace=True)
    return df


def get_aggregated_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    returns aggregated (by average per frame) values of hue,
    lightness, saturation and mask
    '''
    # lightness series
    ls = df.groupby('frame').lightness.mean()
    # hue hls
    hue = df.groupby('frame').hue.mean()
    # saturtion hls
    saturation = df.groupby('frame').saturation.mean()
    # mask
    mask = df.groupby('frame').masked_values.sum()

    cdf = pd.concat([ls, hue, saturation, mask], axis=1)
    cdf = cdf.assign(
        light_diff=lambda x: x.lightness.shift(1) - x.lightness,
        hue_diff=lambda x: x.hue.shift(1) - x.hue,
        saturation_diff=lambda x: x.saturation.shift(1) - x.saturation,
        mask_diff=lambda x: x.masked_values.shift(1) - x.masked_values
    )

    return cdf


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
        return self.frame[:, :, 1]


@dataclass
class Video:
    _vid: ArrayLike
    fps: float

    @property
    def hue(self) -> ArrayLike:
        return self._vid[:, :, :, 0]

    @property
    def saturation(self) -> ArrayLike:
        return self._vid[:, :, :, 2]

    @property
    def lightness(self) -> ArrayLike:
        return self._vid[:, :, :, :1]

    @property
    def width(self):
        return self._vid.shape[2]

    @property
    def height(self):
        return self._vid.shape[1]

    @property
    def frame_count(self):
        return self._vid.shape[0]

    @property
    def arr(self):
        return self._vid

    @classmethod
    def from_file(cls, path: str, conversion: int = cv.COLOR_BGR2HLS) -> Self:
        vid, fps = VideoReader.get_vid(path, conversion)
        return cls(vid, fps)

    def __getitem__(self, frame_no: Union[int, slice]) -> Union[Frame, Self]:
        if isinstance(frame_no, int):
            frame = self._vid[frame_no]
            seconds = np.float128(frame_no / self.fps)
            return Frame(frame, frame_no, seconds)
        start, stop, step = frame_no.indices(self._vid.shape[0])
        return Video(self._vid[start:stop:step], self.fps)

    def __setitem__(self, frame_no: int, new_val: int):
        if new_val > 255:
            raise Exception(
                "255 is too great a value to be represented with np.uint8")
        self._vid[frame_no] = new_val

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
        arr = self._vid.copy()
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
        for index, frame in enumerate(self._vid):
            x = self.width * (index % n_width)
            y = frame.shape[0] * (index // n_width)
            color_correct = cv.cvtColor(frame, cv.COLOR_HLS2RGB)
            img = Image.fromarray(color_correct)
            ret_img.paste(img, (x, y, x + self.width, y + self.height))
        return ret_img
