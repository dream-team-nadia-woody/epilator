from typing import Tuple, Union
import cv2 as cv
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from dataclasses import dataclass
from collections import deque


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
        #frame = cv.resize(frame, dsize=(600, 400))
        return frame


def get_video(path: str) -> Tuple[NDArray, int]:
    vid = Video(path)
    fps = vid.video.get(cv.CAP_PROP_FPS)
    video = np.fromiter(vid, np.ndarray)
    return np.stack(video), int(round(fps))


def get_vid_df(path: str) -> pd.DataFrame:
    vid,fps = get_video(path)
    frames = vid.shape[0]
    height = vid.shape[1]
    width = vid.shape[2]
    df = pd.DataFrame(vid.reshape((-1,3)))
    df['frame'] = df.index // (width * height)
    df['x'] = df.index % width
    df['y'] = df.index // height
    df = df.set_index(['frame','y','x']).rename(columns={
        0:'hue',
        1:'saturation',
        2:'value',
    })
    return df,fps