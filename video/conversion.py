import cv2 as cv
from enum import Enum
from dataclasses import dataclass
from typing import Union
import numpy as np


@dataclass
class Converter:
    load: int
    display: int


class Conversions(Enum):
    HLS = Converter(cv.COLOR_BGR2HLS, cv.COLOR_HLS2RGB)
    HSV = Converter(cv.COLOR_BGR2HSV, cv.COLOR_HSV2RGB)
    RGB = Converter(cv.COLOR_BGR2RGB, 0)
    BGR = Converter(0, cv.COLOR_BGR2RGB)
