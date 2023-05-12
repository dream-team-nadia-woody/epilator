import pandas as pd
import numpy as np
import video.df as df
from video import vid
from video import df
from video import frame
#from video import preprocess as pr

import video.reader as r
import cv2 as cv
from video.reader import VideoReader


from typing import List, Union
from numpy.typing import ArrayLike


######## HELPER FUNCTIONS #########

# FORWARD FILL 
def ffill(arr: np.array):
    '''forward fill the values in 1D numpy array'''
    # create a mask for null values
    mask = np.isnan(arr)
    # create an array to hold indexes
    idx = np.where(~mask, np.arange(mask.size), 0)
    # forward fill indexes (accumulate will retrun the max # of every element in array)
    np.maximum.accumulate(idx, out=idx)
    # forward fill the original array
    return arr[idx]

# BACKWARD FILL
def bfill(arr: np.array):
    '''backward fill 1D numpy array'''
    # create a reverse mask for null values
    mask = np.isnan(arr[::-1])
    # create an array to hold indexes for reversed array
    idx = np.where(~mask, np.arange(mask.size), 0)
    # backill indexes (forward fill the reversed array of indexes)
    np.maximum.accumulate(idx, out=idx)
    # apply reversed indexes to revered array and reverse it back to get an original array :)
    return arr[::-1][idx][::-1]


######### ZERO CROSSING ###########

'''
Extract lightness difference
Save indexes where change in lightness crosses zero
Calculate where it happens more than 3 times in row (or 3 times per second in 30 frames window?)

'''

#### Get lightness difference from the video
def get_lightness_difference(vid: Union[str, ArrayLike],
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
    # save all values but NaN
    ld = np.diff(av_lightness_per_frame, 1)

    # normalize the array
    #ld = ld/np.linalg.norm(ld)
    # normalize with minmax
    ld = (ld - ld.min())/(ld.max() - ld.min()) - 0.5 
    return ld, fps

def find_zero_crossings(lightness_difference: np.array, threshold:float = 0.02):
    ''' 
    returns an array of indexes in the lightness difference
    where the element changes the sign
    if the 
    '''
    ld = lightness_difference.copy()
    ld = np.where(np.absolute(ld) < threshold, 
                                    np.nan, ld)
    # make ffill and bfill, because if we leave values as nan or 0
    # it will count zero crossing in every location of nan and the next index as well
    ld = ffill(ld)
    ld = bfill(ld)

    return np.where(np.diff(np.sign(ld)))[0] + 1


def find_hazard_crossings_per_second(zero_crossings: np.array, 
                            ld_length: int,
                            crossings_per_second:int = 3, 
                            fps:int =30):
    '''
    Find if there are (3) or more numbers of frames in zero_crossing array per second.
    Parameters:
    sliding_windows: 1D numpy array of zero crossings. 
    Default is 3. 3 consecutive flashes per second are dangerous
    Returns:
    list of indexes that start flashes
    '''
    
    # for every slide in sliding windows of size FPS
    # create sliding windows of size 3
    if len(zero_crossings) <= crossings_per_second:
        return np.empty(0, dtype=int)
    else:
        windows = np.lib.stride_tricks.sliding_window_view(np.arange(ld_length), \
                                    window_shape=fps)
        # create an array to hold hazard frames
        hf = np.empty(0, dtype=int)
        for window in windows:
            # find frames that are the same in the sliding window and zero_crossing
            a = np.intersect1d(window, zero_crossings)
            if len(a) > 3 and len(a) < 55:
                hf = np.append(hf, a)
    
        # return list of seconds in video that are hazard
        return np.unique(hf)

def frames_to_seconds(frame_numbers: np.array, fps):
    ''' 
    takes a frame numbers
    returns a list of seconds in the video where the content can cause a seizure
    # '''
    # if frame_numbers == 0 or fps == 0:
    #     return 0
    # else:
    seconds = frame_numbers // fps
    return list(set(seconds))

def run_lightness_test(path: str, crossings_per_second:int = 3, threshold: float = 0.02):
    ld, fps = get_lightness_difference(path)
    zero_crossings = find_zero_crossings(ld, threshold)
    #hc_frames = find_hazard_crossings(zero_crossings)
    hc_frames = find_hazard_crossings_per_second(zero_crossings, len(ld), crossings_per_second, fps)
    return frames_to_seconds(hc_frames, fps)


