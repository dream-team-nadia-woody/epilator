import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, Callable, List


def graph_frames(df: pd.DataFrame, key: Union[str, List[str]],
                 func: Callable,
                 pct_change=0,
                 by_seconds: bool = True) -> plt.Axes:
    grouped_frame = func(df.groupby('frame'), key)
    ax = grouped_frame.plot.line()
    if by_seconds:
        ax.set_xticklabels([tick // df.attrs['fps']
                           for tick in ax.get_xticks()])
        ax.set_xlabel('Seconds')
    ax.set_xlim(0, df.index.levels[0].shape[0])
    return ax


def graph_all_channels(df: pd.DataFrame,
                       func: Callable,
                       colors: List[str] = ['red', 'green', 'blue'], **kwargs):
    fig, axs = plt.subplots(3, 1, **kwargs)
    groupby = df.groupby('frame')
    for ax, channel, color in zip(axs,df.columns, colors):
        ax = graph_frames()
