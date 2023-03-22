import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import cv2 as cv

import acquire as ac
import video.vid as v

import warnings
warnings.filterwarnings("ignore")

def  frame_vs_sec(df):
    '''
    plot lightness grouped by frame and grouped by second
    '''
    fps = df.attrs['fps']

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


def frame_diff_changes(df):
    '''
    Create an aggragated data frame, 
    plots values per frame and difference in values per frame
    '''

    cdf = v.get_aggregated_df(df)
    fps = df.attrs['fps']

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