import cv2 as cv
from enum import Enum
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Converter:
    load: int
    display: int
    channel_names: List[str]


class Conversions(Enum):
    HLS = Converter(cv.COLOR_BGR2HLS, cv.COLOR_HLS2RGB,
                    ['hue', 'lightness', 'saturation'])
    HSV = Converter(cv.COLOR_BGR2HSV, cv.COLOR_HSV2RGB,
                    ['hue', 'saturation', 'value'])
    RGB = Converter(cv.COLOR_BGR2RGB, 0, ['red', 'green', 'blue'])
    BGR = Converter(0, cv.COLOR_BGR2RGB, ['blue', 'green', 'red'])
