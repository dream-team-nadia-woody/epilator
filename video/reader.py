import cv2 as cv
from numpy.typing import ArrayLike
from typing import List, Tuple, Union
import numpy as np
from video.conversion import Conversions, Converter
import os
FRAME_X = 100
FRAME_Y = 100


class VideoReader:
    '''A class which reads in the video from file'''
    video: cv.VideoCapture

    def __init__(self, path: str):
        '''# Parameters:
        path: the relative path of the video file to be read in.'''
        self.video = cv.VideoCapture(path)

    def __del__(self):
        '''Releases the `VideoCapture` object (frees memory)'''
        self.video.release()

    def __iter__(self):
        '''iterates over the video while open'''
        if not self.video.isOpened():
            raise StopIteration
        return self

    def __next__(self):
        '''Gets the next frame of the video'''
        end, frame = self._video.read()
        if not end:
            raise StopIteration
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
        frame = cv.resize(frame, (FRAME_X, FRAME_Y),
                          interpolation=cv.INTER_NEAREST)
        return frame

    @classmethod
    def get_vid(cls, path: str,
                conversion: int
                ) -> Tuple[ArrayLike, float]:
        '''
        Loads the video file into memory
        ## Parameters:
        path: string with the path to the file to be read in
        conversion: OpenCV constant indicating the color
        conversion to be performed on the file
        ## Returns:
        A `tuple` object containing:
            - the video at `path` as a `np.ndarray`
            - the video speed in frames per second
        '''
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