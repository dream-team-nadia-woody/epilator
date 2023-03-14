import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, Callable, List


def graph_frames(df: pd.DataFrame, key: Union[str, List[str]],
                 agg: Union[str,
                            Callable,
                            List[Union[
                                str,
                                Callable]]],
                                pct_change = 0,
                                by_seconds:bool = True) -> plt.Axes:
    grouped_frame = df.groupby('frame')[key].agg(agg)
    if pct_change > 0:
        grouped_frame = grouped_frame.pct_change(pct_change)
    ax = grouped_frame.plot.line()
    ax.set_xticklabels([tick // 30 for tick in ax.get_xticks()])
    if by_seconds:
        ax.set_xlabel('Frames')
    return ax
