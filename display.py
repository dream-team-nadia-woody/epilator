from typing import Iterator, Union

import pandas as pd
from cv2 import cvtColor, COLOR_HSV2RGB as hsv_rgb
from PIL import Image
from numpy.typing import ArrayLike
import numpy as np
from video import FRAME_X, FRAME_Y, Video
import os


def create_image(arr: Union[pd.DataFrame, ArrayLike],
                 frame_x: int = FRAME_X,
                 frame_y: int = FRAME_Y) -> Image:
    if isinstance(arr, pd.DataFrame):
        arr = arr.to_numpy().reshape((frame_x, frame_y, 3))
    arr = cvtColor(arr, hsv_rgb)
    return Image.fromarray(arr)


def show_frame(df: pd.DataFrame,
               frames: Union[int, Iterator], width: int = 5) -> Image:
    frame_x = df.index.get_level_values('x').max() + 1
    frame_y = df.index.get_level_values('y').max() + 1
    if isinstance(frames, int):
        frame = df[df.index.get_level_values('frame') == frames]
        return create_image(frame, frame_x, frame_y)
    all_frames = [create_image(df[df.index.get_level_values(
        'frame') == frame], frame_x, frame_y) for frame in frames]
    ret_width = frame_x * width
    ret_height = frame_x * ((len(all_frames))//width)
    print(ret_width, ret_height, frame_x, frame_y)
    ret_img = Image.new('RGB', (ret_width, ret_height))
    for index, frame in enumerate(all_frames):
        x = frame_x * (index % width)
        y = frame_y * (index // width)
        ret_img.paste(frame, (x, y, x + frame_x, y + frame_y))
    return ret_img


def show_sequence(arr: Union[pd.DataFrame,
                             ArrayLike, Video], n: int = 5) -> Image:
    if isinstance(arr, pd.DataFrame):
        arr = arr.to_numpy().reshape(
            (-1, FRAME_Y, FRAME_X, 3))
    elif isinstance(arr, Video):
        arr = arr.arr
    if arr.shape[0] < n:
        ret_width = arr.shape[0] * arr.shape[2]
        ret_height = arr.shape[1]
    else:
        ret_width = arr.shape[2] * n
        ret_height = arr.shape[0] * arr.shape[1] // n
    final_image = Image.new('RGB', (ret_width, ret_height))
    for index, img in enumerate(arr):
        x = img.shape[1] * (index % n)
        y = img.shape[0] * (index // n)
        final_image.paste(create_image(
            img), (x, y, x+img.shape[1], y + img.shape[0]))
    return final_image
