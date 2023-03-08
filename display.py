from typing import Iterator, Union

import pandas as pd
from cv2 import cvtColor, COLOR_HLS2RGB as hls_rgb
from PIL import Image
from numpy.typing import NDArray
import numpy as np


def create_image(arr: pd.DataFrame, frame_x: int, frame_y: int) -> Image:
    arr = arr.to_numpy().reshape((frame_x, frame_y, 3))
    arr = cvtColor(arr, hls_rgb)
    return Image.fromarray(arr)


def show_frame(df: pd.DataFrame, frames: Union[int, Iterator], width: int = 5) -> Image:
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
