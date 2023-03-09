import cv2
import numpy as np 
import os

def get_mask(img: np.array):
    '''
    get the lightness mask
    '''

    Lchannel = img[:,:,1]
    mask = cv2.inRange(Lchannel, 160, 255)
    result = cv2.bitwise_and(blue,blue, mask= mask)

    return result

def plot_masked_img(img: np.array):
    '''
    plot the image
    '''
    img = cv2.