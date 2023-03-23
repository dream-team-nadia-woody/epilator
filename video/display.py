from typing import Iterator, Union

import pandas as pd
from cv2 import cvtColor, COLOR_HSV2RGB as hsv_rgb
from PIL import Image
from numpy.typing import ArrayLike
import numpy as np
from video.reader import FRAME_X, FRAME_Y
import os
from video.vid import Video
from warnings import warn

warn('''This module is deprecated.
Please use `Video.show()` to achieve the same outcome''')
def create_image(arr: Union[pd.DataFrame, ArrayLike],
                 frame_x: int = FRAME_X,
                 frame_y: int = FRAME_Y) -> Image:
    if isinstance(arr, pd.DataFrame):
        arr = arr.to_numpy().reshape((frame_x, frame_y, 3))
    arr = cvtColor(arr, hsv_rgb)
    return Image.fromarray(arr)

def show_sequence(arr: Union[pd.DataFrame,
                             ArrayLike, Video], n: int = 5) -> Image:
    '''
    Shows a sequence of frames from a `Video` `np.ndarray` or `pd.DataFrame`
    ## Parameters:
    arr: Either a `Video`, `np.ndarray` or `pd.DataFrame` 
    containing the video to be displayed
    n: the number of images to display horizontally
    ## Returns:
    a `PIL.Image` object of dimensions :
    
    `(n * [frame width], [frame height] * [frame count] // n)`
    '''
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
