import numpy as np
import video.df as df
from video import vid


import cv2 as cv
from video.reader import VideoReader

import re
import os
import pickle

from typing import List, Union
from numpy.typing import ArrayLike

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import tensorflow as tf

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

def get_lightness_difference(vid: Union[str, ArrayLike], fps: int = 30,
               conversion: int = cv.COLOR_BGR2HLS) -> np.array:
    '''
    returns an numpy array of difference in mean of lightness between frames.
    the length of the array = number of video frames - 1
    '''
    if isinstance(vid, str):
        vid, fps = VideoReader.get_vid(vid, conversion)
    frames = vid.shape[0]
    height = vid.shape[1]
    width = vid.shape[2]
    # creates an numpy array with the lightness values of each frame
    # the shape of the array is (frames, height*width) f.e. (300, 2500)
    lightness_per_frame = vid.reshape(-1, 3)[:, 1].reshape((-1, height*width))
    # agg function
    av_lightness_per_frame = np.mean(lightness_per_frame,axis=1)
    # shift values by 1 position down
    shifed_lightness = np.concatenate([np.zeros(1), av_lightness_per_frame[:-1]])
    # set 1st value to NaN
    shifed_lightness[0] = np.nan
    # get the difference in lightness between frames
    diff_lightness = shifed_lightness - av_lightness_per_frame
    # return all values but NaN
    return diff_lightness[1:]

def get_file_names(directory:str):
    '''
    returns the list of video files in the directory
    '''
    # get the list of filenames in the directory
    filenames = \
        [f for f in os.listdir(directory) 
        if (os.path.isfile(os.path.join(directory, f)) and f.endswith('.mp4'))]
    return filenames

def get_sequence(directory: str):
    '''
    Goes through every video file in the directory, calculates the lightness difference and
    saves it into a dictionary where:
    Sequence is the list of light differences for each video
    Class name -> hazard / hidden (hazard) / safe
    Saves the result into a pickle file in the directory.
    If this pickle file exists, loads the data from it.
    '''
    # create path to the file
    pickle_file = 'data.pkl'
    pickle_path = directory + '/' + pickle_file
    # load the data if file exists and return it
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as fp:
            return pickle.load(fp)
    else:
        # get names of video files
        filenames = get_file_names(directory)
        # create a dictionary
        data = {'Sequence':[], 'ClassName':[]}
        for f in filenames:
            path = directory + '/' + f
            # read the video and get the lightness difference
            sq = get_lightness_difference(path)
            # get the class name from the file name
            class_name = re.search('(\w+)', f).group()
            # append to the lists in data
            data['Sequence'].append(sq)
            data['ClassName'].append(class_name)
        # save data into a pickle file
        with open(pickle_path, "wb") as outfile1:
            pickle.dump(data, outfile1)
        return data

def get_max_length(vectors:Union[list, np.array]):
    return max(len(v) for v in vectors)

def single_video_padding(light_diff: Union[list, np.array], 
                        max_len: int,
                        pad:str='reflect'):
    ''' '''
    return np.pad(light_diff, (0, max_len - len(light_diff)), pad)

def create_padding(vectors:Union[list, np.array], pad:str='reflect'):
    ''' 
    Used in the next function preprocess_ann()

    for the lightness difference creates a padding.
    max length in the train set is 7000+ frames
    min is 150. 
    creates all lists in the data dictionary with length of max length
    pad = 'reflect' repeats the values of the sequence
    pad = 'constant' will replace the missing values with zeros
    '''
    # load the values into vectors var
    #vectors = np.array(sq)

    # Determine the maximum length of the vectors
    max_len = max(len(v) for v in vectors)

    # Pad the vectors
    padded_vectors = []
    for v in vectors:
        padded_vector = np.pad(v, (0, max_len - len(v)), pad)
        padded_vectors.append(padded_vector)

    # Create a mask
    # mask = []
    # for v in padded_vectors:
    #     mask.append([1] * len(v) + [0] * (max_len - len(v)))

    # Convert the lists to numpy arrays
    padded_vectors = np.array(padded_vectors)
    # mask = np.array(mask)

    return padded_vectors

def scaler(X_train):
    ''' 
    return Standard Scaler ready for transformations
    '''
    sc = StandardScaler()
    X_train = sc.fit(X_train)
    return sc

def preprocess_ann(data:dict):
    ''' 
    preprocess the data for ANN
    no test set at the moment!!!
    need to download more training/test videos

    returns X_train and y_train ready to feed to ANN (only)
    ''' 
    data['Sequence'] = create_padding(data['Sequence'])
    # np.array just in case :)
    X_train = np.array(data['Sequence'])
    y_train = np.array(data['ClassName'])
    # scale with Standard Scaler
    sc = scaler(X_train)
    X_train = sc.transform(X_train)
    # create numerical values for the categorical labels
    le = LabelEncoder()
    num_labels = le.fit_transform(y_train)
    # make sure that the label array is of the same data type as the numpy array
    y_train = num_labels.astype(X_train.dtype)
    # create labels on the shape (20, 3) where
    # [1., 0., 0.] - 'hazard'
    # [0., 0., 1.] - 'safe' 
    # [0., 1., 0.] - 'hidden hazard'
    y_train = np_utils.to_categorical(y_train)

    return X_train, y_train


###### GLOBAL VARS
directory = '../videos1'
data = get_sequence(directory=directory)
sq = data['Sequence']
X_train, y_train = preprocess_ann(data)
sc = scaler(X_train)
max_len = get_max_length(sq)

def single_video_preprocess(path:str):
    ''' '''
    v = get_lightness_difference(path)
    v = single_video_padding(v, max_len)
    v = sc.transform(v.reshape(1,-1))

    return v