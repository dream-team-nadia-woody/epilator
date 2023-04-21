import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike
from PIL import Image

from video.conversion import Converter
from video.videolike import VideoLike


class Frame(VideoLike):
    '''
    A wrapper class representing the single frame of
    a video. This prevents issues with reshaping
    of the array.
    '''
    conversion: Converter
    fps: float
    frame_no: np.uint64
    seconds: np.float128

    def __init__(self, frame: ArrayLike, fps: float, conversion: Converter,
                 frame_no: np.uint64) -> None:
        super().__init__(frame, None, conversion)
        self.frame_no = frame_no
        self.seconds = np.float128(frame / fps)
        self.fps = fps

    def show(self) -> Image.Image:
        '''
        Returns the frame as a PIL Image object
        ## Parameters:
        None
        ## Returns:
        a PIL Image object
        '''
        return self._get_img(self.vid)
    
    def segment(self,segments:int):
        pass