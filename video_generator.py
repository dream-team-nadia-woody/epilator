import cv2 as cv
import numpy as np
from typing import Callable, Union
from numpy.typing import NDArray

fourcc = cv.VideoWriter_fourcc(*'mp4v')
FRAME_WIDTH = 1900
FRAME_HEIGHT = 1080
FRAME_FPS = 30


def black_frame(): return np.zeros(
    (FRAME_HEIGHT,
     FRAME_WIDTH,
     3),
    dtype=np.uint8)


def vid_frame(frames): return np.zeros(
    (frames, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def create_vid(path: str, frames: Union[NDArray, Callable]):
    writer = cv.VideoWriter(path, fourcc,
                            FRAME_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return


def create_black_white_flash() -> NDArray:
    frames = 10 * FRAME_FPS
    white_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 255, dtype=np.uint8)
    ret_arr = np.zeros((frames, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    for i in range(0, frames, 3):
        ret_arr[i] = white_frame
    return create_vid('videos/Black White Flash.mp4', ret_arr)


def create_spinning_red():
    frames = 10 * FRAME_FPS
    ret_arr = vid_frame(frames)
    width = ret_arr.shape[2] // 2
    height = ret_arr.shape[1] // 2
    for index, frame in enumerate(ret_arr):
        i = index % 4
        x, y = 0, 0
        if i == 1:
            x = width
        elif i == 2:
            x = width
            y = height
        elif i == 3:
            y = height
        cv.rectangle(frame, (x,y),(x+width, y + height),(0,0,255),-1)
    return create_vid('videos/Red Spin.mp4', ret_arr)

def blue_green_fade():
    ret_frames = vid_frame(256 * 3)
    for i in range(256):
        ret_frames[i,:,:,0] = i
    ret_frames[256:512,:,:,0] = 255
    for i in range(256):
        ret_frames[256+i,:,:,1] = i
    ret_frames[512:,:,:,1] = 255
    for i in range(256):
        ret_frames[512+i,:,:,0] = 255-i
    return create_vid('videos/Blue Green Fade.mp4',ret_frames)



if __name__ == "__main__":
    create_black_white_flash()
    create_spinning_red()
    blue_green_fade()
