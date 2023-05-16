import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import cv2 as cv

from typing import Union

import acquire as ac
import vid as v
import df as df

import warnings
warnings.filterwarnings("ignore")

def  frame_vs_sec(adf):
    '''
    plot lightness grouped by frame and grouped by second
    '''
    fps = adf.attrs['fps']

    plt.figure(figsize=(14,5))
    plt.subplot(121)
    ax1 = df.groupby('frame').lightness.mean().plot()
    plt.title('Lightness per frame')
    ax1.set_xticklabels(ax1.get_xticks() // fps)
    ax1.set_xlabel('seconds')
    plt.subplot(122)
    df.groupby('seconds').lightness.mean().plot()
    plt.title('Lightness per second')
    plt.show()


def frame_diff_changes(adf):
    '''
    Create an aggragated data frame, 
    plots values per frame and difference in values per frame
    '''

    cdf = df.get_aggregated_df(adf)
    fps = adf.attrs['fps']

    plt.figure(figsize=(20, 20))
    plt.suptitle('Values and Difference in Values')

    plt.subplot(421)
    cdf.lightness.plot()
    plt.title('Lightness per Frame')

    plt.subplot(422)
    cdf.light_diff.plot()
    plt.title('Difference in lightness')

    plt.subplot(423)
    cdf.hue.plot()
    plt.title('Hue Values per Frame')

    plt.subplot(424)
    cdf.hue_diff.plot()
    plt.title('Difference in Hue')

    plt.subplot(425)
    cdf.saturation.plot()
    plt.title('Saturation per Frame')

    plt.subplot(426)
    cdf.saturation_diff.plot()
    plt.title('Difference in Saturation')

    plt.subplot(427)
    ax1 = cdf.masked_values.plot()
    plt.title('Masked light per Frame')
    ax1.set_xticklabels(ax1.get_xticks() // fps)
    ax1.set_xlabel('seconds')

    plt.subplot(428)
    ax2 = cdf.mask_diff.plot()
    plt.title('Difference in Masked Light')
    ax2.set_xticklabels(ax1.get_xticks() // fps)
    ax2.set_xlabel('seconds')

    plt.show()

def fourier_features(ser: Union[pd.DataFrame, np.array], 
                    freq: int =30, 
                    order: int =4) -> pd.DataFrame:
    '''
    creates fourier features.
    ser: pandas series or 1D np.array
    freq: frequency
    order: the number of sin/cos waves

    '''
    time = np.arange(len(ser), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.DataFrame(features, index=ser)

def add_fourier_features(df: pd.DataFrame(), 
                            col_name: str, 
                            freq:int=30, 
                            order:int=4) -> pd.DataFrame:
    '''
    creates fourier features and adds them to data frame.
    ser: pandas series or 1D np.array
    freq: frequency
    order: the number of sin/cos waves

    '''
    time = np.arange(len(df), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"{col_name}_sin_{freq}_{i}": np.sin(i * k),
            f"{col_name}_cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)