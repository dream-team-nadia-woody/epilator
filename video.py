from typing import Tuple, Union
import cv2 as cv
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from collections import deque

FRAME_X = 100
FRAME_Y = 100


class Video:
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

    def get_vid(self, conversion: int) -> NDArray:
        if not self.video.isOpened():
            raise EOFError('File not opened')
        frames = []
        fps = self.video.get(cv.CAP_PROP_FPS)
        while True:
            flag, frame = self.video.read()
            if not flag:
                break
            if conversion > 0:
                frame = cv.cvtColor(frame, conversion)
            frame = cv.resize(frame, (FRAME_X, FRAME_Y),
                              interpolation=cv.INTER_NEAREST)
            frames.append(frame)
        return np.stack(np.asarray(frames)),int(round(fps))


def get_video_from_iterator(path: str) -> Tuple[NDArray, int]:
    vid = Video(path)
    fps = vid.video.get(cv.CAP_PROP_FPS)
    video = np.fromiter(vid, np.ndarray)
    return np.stack(video), int(round(fps))


def get_vid_df(vid: Union[str, NDArray], fps: int = 30, conversion: int = cv.COLOR_BGR2HSV) -> pd.DataFrame:
    if isinstance(vid, str):
        vid, fps = Video(vid).get_vid(conversion)
    frames = vid.shape[0]
    height = vid.shape[1]
    width = vid.shape[2]
    df = pd.DataFrame(vid.reshape((-1, 3)))
    df['frame'] = df.index // (width * height)
    df['x'] = df.index % width
    df['y'] = df.index // width % height
    df = df.set_index(['frame', 'y', 'x']).rename(columns={
        0: 'hue',
        1: 'saturation',
        2: 'value',
    })
    df.attrs['fps'] = fps
    df.attrs['conversion'] = conversion
    return df
