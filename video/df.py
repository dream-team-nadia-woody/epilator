

from typing import List, Union
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import cv2 as cv
from video.reader import VideoReader

import os
import re
import pickle


def get_vid_df(vid: Union[str, ArrayLike], fps: int = 30,
               conversion: int = cv.COLOR_BGR2HLS,
               rename: List[str] = [
        'hue', 'lightness', 'saturation']) -> pd.DataFrame:
    '''
    Loads a video into a Pandas `DataFrame`
    ## Parameters:
    vid: either a `np.ndarray` object containing containing the target video or a string with its path
    fps: the speed of the video in frames per second
    conversion: the OpenCV color conversion constant with which to convert the video (default is HLS).
    rename: a `list` to rename the numeric columns to in the finished `DataFrame`.
    ## Returns:
    A DataFrame, indexed on the frame and the x and y coordinates of the corresponding pixel, as well
    as its three color values.
    '''
    if isinstance(vid, str):
        vid, fps = VideoReader.get_vid(vid, conversion)
    frames = vid.shape[0]
    height = vid.shape[1]
    width = vid.shape[2]
    df = pd.DataFrame(vid.reshape((-1, 3)))
    df['frame'] = df.index // (width * height)
    df['x'] = df.index % width
    df['y'] = df.index // width % height
    rename = {key: value for key, value in enumerate(rename)}
    df = df.set_index(['frame', 'y', 'x']).rename(columns=rename)

    # to call attributes -> df.attrs['width']
    df.attrs['height'] = height
    df.attrs['width'] = width
    df.attrs['fps'] = fps
    df.attrs['conversion'] = conversion

    return df


def get_mask(img: np.array):
    '''
    get the lightness mask from hls image
    '''

    Lchannel = img[:, :, 1]
    mask = cv.inRange(Lchannel, 160, 255)
    # mask = np.where(255, 1, 0)

    return mask

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


def add_mask(df: pd.DataFrame) -> pd.DataFrame:
    '''
    calls get_frame() to generate mask for each frame. s
    aves results in np.array of 1 and 0
    where 1 - light pixel, 0 - dark pixel
    '''
    # let's try the same but through numpy array
    narr = np.empty((0,), dtype=np.uint8)
    w = df.attrs['width']
    h = df.attrs['height']
    # loop through the frames
    for f in df.index.levels[0]:
        # save an image to the variable 'frame'
        frame = df.loc[f].iloc[:, :].to_numpy().reshape(w, h, 3)
        # get the mask
        mask = get_mask(frame)
        narr = np.concatenate([narr, mask.reshape(-1)])
    return df.assign(masked_values=narr)

def add_red_mask(df: pd.DataFrame) -> pd.DataFrame:
    '''
    calls get_frame() to generate mask for each frame. s
    aves results in np.array of 1 and 0
    where 1 - light pixel, 0 - dark pixel
    '''
    # let's try the same but through numpy array
    narr = np.empty((0,), dtype=np.uint8)
    w = df.attrs['width']
    h = df.attrs['height']
    # loop through the frames
    for f in df.index.levels[0]:
        # save an image to the variable 'frame'
        frame = df.loc[f].loc[:, ['hue', 'lightness', 'saturation']].to_numpy().reshape(w, h, 3)
        # get the mask
        mask = get_red_mask(frame)
        narr = np.concatenate([narr, mask.reshape(-1)])
    return df.assign(red_values=narr)


def add_seconds(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Adds columns 'seconds' that shows the second of the video
    '''
    return df.assign(
        seconds=lambda x: x.index.get_level_values('frame') // x.attrs['fps']
    )


def get_exploration_df(vid: Union[str, ArrayLike], fps: int = 30,
                       conversion: int = cv.COLOR_BGR2HLS) -> pd.DataFrame:
    '''
    returns a data frame of the video with mask values and seconds added
    '''
    df = get_vid_df(vid)
    df = add_mask(df)
    df = add_red_mask(df)
    df = add_seconds(df)
    df.masked_values.replace({255: 1}, inplace=True)
    df.red_values.replace({255: 1}, inplace=True)
    return df


def get_aggregated_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    returns aggregated (by average per frame) values of hue,
    lightness, saturation and mask
    '''
    # lightness series
    ls = df.groupby('frame').lightness.mean()
    # hue hls
    hue = df.groupby('frame').hue.mean()
    # saturtion hls
    saturation = df.groupby('frame').saturation.mean()
    # mask
    mask = df.groupby('frame').masked_values.sum()
    # red values
    red = df.groupby('frame').red_values.sum()

    cdf = pd.concat([ls, hue, saturation, mask, red], axis=1)
    cdf = cdf.assign(
        light_diff=lambda x: x.lightness.shift(1) - x.lightness,
        hue_diff=lambda x: x.hue.shift(1) - x.hue,
        saturation_diff=lambda x: x.saturation.shift(1) - x.saturation,
        mask_diff=lambda x: x.masked_values.shift(1) - x.masked_values
    )
    # get the width and the height of the frames
    # width = len(df.index.levels[2])
    # height = len(df.index.levels[1])

    return cdf
