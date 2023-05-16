import numpy as np
from numpy.typing import NDArray
from video.vid import Video
from enum import Enum
import matplotlib.pyplot as plt
from typing import Callable, Any, Tuple
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster


def get_peaks(agg_vid: NDArray, **peak_args) -> NDArray:
    ''' returns the peaks and valleys of a video'''
    peaks, _ = find_peaks(agg_vid, **peak_args)
    valleys, _ = find_peaks(-agg_vid, **peak_args)
    return np.sort(np.concatenate((peaks, valleys)))


def generate_elbow(peaks: NDArray) -> Tuple[NDArray, plt.Axes]:
    distance_matrix = np.abs(peaks[:, np.newaxis] - peaks)
    link = linkage(distance_matrix,method='ward')
    last = link[-10:,2]
    last_rev = last[::-1]
    idx =np.arange(1,len(last)+1)
    fig, axs = plt.subplots()
    axs.plot(idx, last_rev)
    return link, fig
