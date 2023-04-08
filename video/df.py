

from typing import List, Union
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
import cv2 as cv
from video.reader import VideoReader


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
    df = add_seconds(df)
    df.masked_values.replace({255: 1}, inplace=True)
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

    cdf = pd.concat([ls, hue, saturation, mask], axis=1)
    cdf = cdf.assign(
        light_diff=lambda x: x.lightness.shift(1) - x.lightness,
        hue_diff=lambda x: x.hue.shift(1) - x.hue,
        saturation_diff=lambda x: x.saturation.shift(1) - x.saturation,
        mask_diff=lambda x: x.masked_values.shift(1) - x.masked_values
    )

    return cdf

def get_lightness_difference(vid: Union[str, ArrayLike], fps: int = 30,
               conversion: int = cv.COLOR_BGR2HLS) -> np.array:
    '''
    returns an numpy array of difference in mean of lightness between frames.
    the length of the array = number of video frames - 1
    '''
    if isinstance(vid, str):
        vid, fps = VideoReader.get_vid(vid, conversion)
    frames = vid.shape[0]
    height = vid.shape[1]
    width = vid.shape[2]
    # creates an numpy array with the lightness values of each frame
    # the shape of the array is (frames, height*width) f.e. (300, 2500)
    lightness_per_frame = vid.reshape(-1, 3)[:, 1].reshape((-1, height*width))
    # agg function
    av_lightness_per_frame = np.mean(lightness_per_frame,axis=1)
    # shift values by 1 position down
    shifed_lightness = np.concatenate([np.zeros(1), av_lightness_per_frame[:-1]])
    # set 1st value to NaN
    shifed_lightness[0] = np.nan
    # get the difference in lightness between frames
    diff_lightness = shifed_lightness - av_lightness_per_frame
    # return all values but NaN
    return diff_lightness[1:]
