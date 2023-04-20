from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Self, Tuple, Union, Literal
from warnings import warn

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from PIL import Image
from io import BytesIO
import os
from typing_extensions import override

from video.channel import AGG_FUNCS, AggregatorFunc, Channel
from video.conversion import Conversions, Converter
from video.frame import Frame
from video.videolike import VideoLike
from IPython.display import display, Image as IPyImage


class Video(VideoLike):
    '''A Class to store videos'''
    fps: float
    converter: Converter
    start_time: np.float128
    end_time: np.float128

    def __init__(self, vid: Union[ArrayLike, Self],
                 fps: Union[float, None] = None,
                 converter: Conversions = Conversions.HLS,
                 start_time: Union[float, None] = None,
                 end_time: Union[float, None] = None,
                 segments: int = 1) -> None:
        '''
        Creates a new video object
        ## Parameters:
        vid: either an `np.ndarray` or `Video` object
        fps: the frames per second of the video
        '''
        if isinstance(vid, Video):
            fps = vid.fps
            converter = vid.converter
            vid = vid.vid
        super().__init__(vid, fps, converter, segments)
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = vid.shape[0] / fps
        self.start_time = start_time
        self.end_time = end_time

    @property
    def frame_count(self) -> int:
        '''the number of frames in the video'''
        return self.shape[0]

    @property
    def length(self) -> np.float128:
        return np.float128(self.end_time - self.start_time)
    
    @property
    def aspect_ratio(self)->float:
        '''returns the ratio of width to height'''
        return self.width / self.height
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
            frame = self.vid[frame_no]
            seconds = np.float128(frame_no / self.fps)
            return Frame(frame, self.fps, self.converter, frame_no)
        start, stop, step = frame_no.indices(self.vid.shape[0])
        clip_start = self.start_time + (start / self.fps)
        clip_end = self.end_time + (stop / self.fps)
        return Video(self.vid[start:stop:step], self.fps, self.converter, clip_start, clip_end)

    def __setitem__(self, frame_no: int, new_val: int):
        if new_val > 255:
            raise Exception(
                "255 is too great a value to be represented with np.uint8")
        self.vid[frame_no] = np.full(
            (self.height, self.width, 3), new_val, dtype=np.uint8)

    def __get_img(self, frame: ArrayLike) -> Image:
        if self.converter.display > 0:
            frame = cv.cvtColor(frame, self.converter.display)
        return Image.fromarray(frame)

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
        for index, frame in enumerate(self.vid):
            x = self.width * (index % n_width)
            y = frame.shape[0] * (index // n_width)
            img = self.__get_img(frame)
            # Resize the image to the expected dimensions
            ret_img.paste(img, (x, y))
        return ret_img

    def show_gif(self,scale:float = 3.0) -> Image:
        ret_width = int(self.width * scale * self.aspect_ratio)
        ret_height = int(self.width * scale)
        # Create a list of PIL Image objects from the NumPy array
        images = [self.__get_img(frame).resize(
            (ret_width, ret_height)
        ) for frame in self.vid]

        # Save the GIF to an in-memory buffer
        buffer = BytesIO()
        images[0].save(buffer, format='GIF', save_all=True,
                       append_images=images[1:], duration=1000//self.fps, loop=0)

        # Display the GIF in the Jupyter notebook
        buffer.seek(0)
        display(IPyImage(data=buffer.getvalue(), format='GIF'))

    @classmethod
    def analyze_vids(cls,
                     conversion: Conversions = Conversions.HSV,
                     agg_method: Union[Literal['pct_change'],
                                       Literal['agg']] = 'agg',
                     agg_func: Union[str, Callable] = AGG_FUNCS['sum'],
                     vid_dir: str = 'videos',
                     mask_channel: Union[str,
                                         int,
                                         None] = None,
                     agg_channel: Union[str,
                                        int,
                                        None] = None) -> Dict[str, np.ndarray]:
        '''
        Goes through a given directory of videos and returns an analysis
        of the videos
        ## Parameters:
        conversion: `Conversions` class to use
        agg_method: the method to use for aggregating the video, either 
        `pct_change` or `agg`
        agg_func: the function to use for aggregating the video, either a string
        in AGG_FUNCS or a custom function
        vid_dir: the directory to look for videos in
        mask_channel: the channel to use for masking the video
        agg_channel: the channel to use for aggregating the video
        ## Returns:
        A dictionary with keys as the name of the video and a value of the `ndarray`
        of the aggregation.
        '''
        videos = os.listdir(vid_dir)
        videos.remove('.DS_Store')
        agg_list = {}
        for index, video in enumerate(videos):
            vid = cls.from_file(os.path.join(vid_dir, video), conversion)
            masked_vid = vid.mask(mask_channel)
            if agg_method == 'pct_change':
                agg_vid = masked_vid.pct_change(1, agg_channel, agg_func)
            else:
                agg_vid = masked_vid.agg(agg_func, agg_channel)
            agg_list[video] = agg_vid
        return agg_list

    def segment(self, n: int) -> Self:
        '''
        Segments a video into n^2 segments
        ## Parameters:
        n: the number of segments to segment the video into
        ## Returns:
        a `Video` object of the segmented video
        '''
        new_vid = self.vid.copy()

        # Calculate new dimensions
        new_width = self.width // n
        new_height = self.height // n
        new_frame_count = self.frame_count * n * n

        # Initialize the new array
        segmented_vid = np.empty(
            (new_frame_count, new_height, new_width, 3), dtype=new_vid.dtype)

        # Segment each frame
        frame_idx = 0
        for frame in new_vid:
            for i in range(n):
                for j in range(n):
                    segment = frame[i * new_height:(i + 1) * new_height,
                                    j * new_width:(j + 1) * new_width, :]
                    segmented_vid[frame_idx] = segment
                    frame_idx += 1

        return Video(segmented_vid, self.fps, self.converter,
                     self.start_time, self.end_time, n)
