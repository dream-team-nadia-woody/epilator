import pandas as pd
import numpy as np
import video.df as df
from video import vid
from video import df
from video import frame
from video import hazard_detection as hd

import video.reader as r
import cv2 as cv
from video.reader import VideoReader


from typing import List, Union
from numpy.typing import ArrayLike


def get_hue_difference(vid: str, conversion=cv.COLOR_BGR2HLS):
    '''
    returns:
    1) an numpy array difference of difference in mean of hue between frames
    the length of the array = number of video frames - 2
    2) fps: frames per second
    '''
    if isinstance(vid, str):
        vid, fps = VideoReader.get_vid(vid, conversion)
    #frames = vid.shape[0]
    height = vid.shape[1]
    width = vid.shape[2]
    # creates an numpy array with the lightness values of each frame
    # the shape of the array is (frames, height*width) f.e. (300, 2500)
    hue_per_frame = vid.reshape(-1, 3)[:, 0].reshape((-1, height*width))
        # agg function
    av_hue_per_frame = np.mean(hue_per_frame,axis=1)
    # save all values but NaN
    hue_d = np.diff(av_hue_per_frame, 1)
    # normalize the array
    #ld = ld/np.linalg.norm(ld)
    # normalize with minmax, -0.5 makes all values between -0.5 and 0.5 with 0 in the middle
    hue_d = (hue_d - hue_d.min())/(hue_d.max() - hue_d.min()) - 0.5 
    # difference of moves all value closer to 0
    return hue_d, fps


def run_lightness_test(path: str):
    hue_d, fps = get_hue_difference(path)
    zero_crossings = hd.find_zero_crossings(hue_d)
    hc_frames = hd.find_hazard_crossings_per_second(zero_crossings, len(hue_d), fps)
    return hd.frames_to_seconds(hc_frames, fps)

