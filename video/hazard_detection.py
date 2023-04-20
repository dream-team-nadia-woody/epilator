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
    seconds = (frame_numbers / fps).astype(int)
    return list(set(seconds))


######## RED LIGHT FLASHES #########
'''
At the moment counts the frames were red color covers more than 25% of the screen
'''

def get_red_mask(img):
    # mask for hue below 10 hue, lightness, saturation
    lb = np.array([0,50,50], dtype=np.uint8)
    ub = np.array([10,255,255], dtype=np.uint8)
    mask1 = cv.inRange(img, lb, ub)

    # # mask for hue above 340
    lb1 = np.array([170,50, 50], dtype=np.uint8)
    ub1 = np.array([180,255, 255], dtype=np.uint8)
    mask2 = cv.inRange(img, lb1, ub1)

    return mask1 | mask2

def detect_red_light(video_path: str, seconds: bool=True):
    '''
    Detects what % of video frames have red color on more than 25% of the screen
    Parameters:
    videao_path: the path to the video to be analyzed

    Returns:
    if seconds: 
        list of seconds where the red light covers more than 25% of the screen
    else: 
        the proportion of the video frames with the red light values covering more than 25% of the screen
    '''
    # get video and fps
    vid, fps = VideoReader.get_vid(video_path, 
                conversion=cv.COLOR_BGR2HLS)
    # get # of frames, width and height of each frame
    width = vid.shape[2]
    height = vid.shape[1]
    frames = vid.shape[0]

    # create en empty array of integers
    narr = np.empty((0,), dtype=np.uint8)
    for v in vid:
        # get video frame
        frame = v.reshape(height, width, 3)
        # apply red mask
        mask = get_red_mask(frame)
        # append masked values as 1D array narr
        narr = np.concatenate([narr, mask.reshape(-1)])
    # narr -> array whith masked red values. represent pixels of each video frame
    # in the video. if the pixel is red it equals 1 else 0
    narr = np.where(narr == 255, 1, 0)
    # count the number of pixels in frame
    total_pixels_per_frame = width * height
    
    # reshape the narr array to the shape of (frame, total pixels)
    narr = narr.reshape(frames, total_pixels_per_frame)

    red_frames = 0
    sec = []
    for n in narr:
        if n.sum() > total_pixels_per_frame / 4:
            red_frames += 1
            sec.append((n / fps).astype(int))
    
    if seconds:
        return sec
    else:
        return round(red_frames / frames * 100, 2)
