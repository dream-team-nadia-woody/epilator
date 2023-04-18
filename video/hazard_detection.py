import pandas as pd
import numpy as np
import video.df as df
from video import vid
from video import df
from video import frame
from video import preprocess as pr

import video.reader as r
import cv2 as cv
from video.reader import VideoReader


from typing import List, Union
from numpy.typing import ArrayLike

######## ZERO CROSSING #########

'''
Extract lightness difference
Save indexes where change in lightness crosses zero
Calculate where it happens more than 3 times in row (or 3 times per second in 30 frames window?)

'''

#### Get lightness difference from the video
def get_lightness_difference(vid: Union[str, ArrayLike], fps: int = 30,
               conversion: int = cv.COLOR_BGR2HLS) -> np.array:
    '''
    returns:
    1) an numpy array of difference in mean of lightness between frames
    the length of the array = number of video frames - 1
    2) fps: frames per second
    '''
    if isinstance(vid, str):
        vid, fps = VideoReader.get_vid(vid, conversion)
    #frames = vid.shape[0]
    height = vid.shape[1]
    width = vid.shape[2]
    # creates an numpy array with the lightness values of each frame
    # the shape of the array is (frames, height*width) f.e. (300, 2500)
    lightness_per_frame = vid.reshape(-1, 3)[:, 1].reshape((-1, height*width))
        # agg function
    av_lightness_per_frame = np.mean(lightness_per_frame,axis=1)
    # return all values but NaN
    return np.diff(av_lightness_per_frame, 1), fps

def find_zero_crossings(lightness_difference: np.array, treshold:int = 10):
    ''' 
    returns an array of indexes in the lightness difference
    where the element changes the sign
    if the 
    '''
    ld = lightness_difference.copy()
    ld = np.where(np.absolute(ld) < treshold, 
                                    0, ld)
    return np.where(np.diff(np.sign(ld)))[0] + 1

def extract_windows(arr: np.array, 
                    start: int, end: int, sub_window_size: int):
    '''
    Creates sliding windows of sub_window_size 
    '''
    
    end = end - sub_window_size + 1
    sub_windows = (
        start +
        # expand_dims are used to convert a 1D array to 2D array.
        np.expand_dims(np.arange(sub_window_size), 0) +
        np.expand_dims(np.arange(end - start), 0).T
    )
    
    # one line function if there is no need for start / end indexes
    # return np.lib.stride_tricks.sliding_window_view(arr, window_shape=sub_window_size)
    return arr[sub_windows]

def find_hazard_crossings_sw(sliding_windows: np.array, consecutive_numbers=3, fps=30):
    '''
    Find if there are (3) consecutive numbers of frames in zero_crossing array.
    Parameters:
    sliding_windows: 2D numpy array of sliding windows of FPS (frames per second) size
    consecutive_numbers: how many consecutive numbers we have to check. 
    Default is 3. 3 consecutive flashes per second are dangerous
    Returns:
    list of indexes that start flashes
    '''
    frame_numbers = np.array([], dtype='int')
    for sw in sliding_windows:
        # for every slide in sliding windows of size FPS
        # create sliding windows of size 3
        windows = np.lib.stride_tricks.sliding_window_view(sw, \
                                    window_shape=consecutive_numbers)
        # check which windows have 3 consectutive numbers 
        cond = np.all(np.diff(windows, axis=1) == 1, axis = 1)
        frame_numbers = np.append(frame_numbers, windows[cond][:, 0])
        # return list of seconds in video that are hazard
    return np.unique(frame_numbers)

'''
replaces 2 functions above
:facepalm:
'''
def find_hazard_crossings(zero_crossings: np.array, consecutive_numbers=3, fps=30):
    '''
    Find if there are (3) consecutive numbers of frames in zero_crossing array.
    Parameters:
    sliding_windows: 1D numpy array of zero crossings. 
    Default is 3. 3 consecutive flashes per second are dangerous
    Returns:
    list of indexes that start flashes
    '''
    
    # for every slide in sliding windows of size FPS
    # create sliding windows of size 3
    windows = np.lib.stride_tricks.sliding_window_view(zero_crossings, \
                                window_shape=consecutive_numbers)
        # check which windows have 3 consectutive numbers 
    cond = np.all(np.diff(windows, axis=1) == 1, axis = 1)
    
    # return list of seconds in video that are hazard
    return np.unique(windows[cond][:, 0])

def frames_to_seconds(frame_numbers: np.array, fps):
    ''' 
    takes a frame numbers
    returns a list of seconds in the video where the content can cause a seizure
    '''
    seconds = frame_numbers / fps.astype(int)
    return list(set(seconds))



