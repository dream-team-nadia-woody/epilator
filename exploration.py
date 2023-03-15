import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import cv2 as cv

import downloader as d 
import video as v

def  frame_vs_sec(df):
    '''
    plot lightness grouped by frame and grouped by second
    '''
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