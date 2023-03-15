import cv2 as cv
import numpy as np
from typing import Callable, Union
from numpy.typing import NDArray

fourcc = cv.VideoWriter_fourcc(*'mp4v')
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_FPS = 30


def black_frame(): return np.zeros(
    (FRAME_HEIGHT,
     FRAME_WIDTH,
     3),
    dtype=np.uint8)


def vid_frame(frames): return np.zeros(
    (frames, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def create_vid(path: str, frames: Union[NDArray, Callable], fps=FRAME_FPS):
    writer = cv.VideoWriter(path, fourcc,
                            fps, (FRAME_WIDTH, FRAME_HEIGHT))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return


def black_white_flash() -> NDArray:
    frames = 10 * FRAME_FPS
    white_frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 255, dtype=np.uint8)
    ret_arr = np.zeros((frames, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    for i in range(0, frames, 3):
        ret_arr[i] = white_frame
    return create_vid('videos/Black White Flash.mp4', ret_arr)


def spinning_red():
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
        cv.rectangle(frame, (x, y), (x+width, y + height), (0, 0, 255), -1)
    return create_vid('videos/Red Spin.mp4', ret_arr)

def blue_green_fade():
    ret_frames = vid_frame(768)
    for i in range(256):
        ret_frames[i, :, :, 0] = i
    ret_frames[256:512, :, :, 0] = 255
    for i in range(256):
        ret_frames[256+i, :, :, 1] = i
    ret_frames[512:, :, :, 1] = 255
    for i in range(256):
        ret_frames[512+i, :, :, 0] = 255-i
    return create_vid('videos/Blue Green Fade.mp4',ret_frames)
def rgb_fade():
    ret_frames = vid_frame(2304)
    for i in range(256):
        ret_frames[i, :, :, 0] = i
    ret_frames[256:512, :, :, 0] = 255
    for i in range(256):
        ret_frames[256+i, :, :, 1] = i
    ret_frames[512:1024, :, :, 1] = 255
    for i in range(256):
        ret_frames[512+i, :, :, 0] = 255-i
    for i in range(256):
        ret_frames[768 + i, :, :, 2] = i
    ret_frames[1024:1792, :, :, 2] = 255
    for i in range(256):
        ret_frames[1024 + i, :, :, 1] = 255-i
    for i in range(256):
        ret_frames[1280 + i, :, :, 0] = i
    for i in range(256):
        ret_frames[1536 + i, :, :, (0, 1)] = 255-i
    ret_frames[1792:2048, :, :, 0] = 255
    for i in range(256):
        ret_frames[1792 + i, :, :, (1, 2)] = i
    ret_frames[2048:, :, :, 1] = 255
    for i in range(256):
        ret_frames[2048+i, :, :, (0, 2)] = 255-i
    return create_vid('videos/Color Fade.mp4', ret_frames, 60)


def blue_green_flash():
    ret_frames = vid_frame(5 * FRAME_FPS)
    for i, frame in enumerate(ret_frames):
        frame[:, :, i % 2] = 255
    return create_vid('videos/Blue Green Flash.mp4', ret_frames)


def color_flash():
    ret_frames = vid_frame(5 * FRAME_FPS)
    for i, frame in enumerate(ret_frames):
        frame[:, :, i % 3] = 255
    return create_vid('videos/Color Flash.mp4', ret_frames)


if __name__ == "__main__":
    black_white_flash()
    spinning_red()
    rgb_fade()
    blue_green_fade()
    blue_green_flash()
    color_flash()
