import cv2 as cv
from numpy.typing import ArrayLike
from typing import Tuple
import numpy as np
FRAME_X = 120
FRAME_Y = 90


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
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
        frame = cv.resize(frame, (FRAME_X, FRAME_Y),
                          interpolation=cv.INTER_NEAREST)
        return frame

    @classmethod
    def get_vid(cls, path: str,
                conversion: int = cv.COLOR_BGR2HLS
                ) -> Tuple[ArrayLike, float]:
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
        return np.stack(frames), int(round(fps))
