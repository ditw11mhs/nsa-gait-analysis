import numpy as np
import pandas as pd
from scipy import signal as sn


def txt_to_df(file):
    data_array = np.loadtxt(file)
    used_data_array = data_array[:, :6]
    dataframe = pd.DataFrame(
        used_data_array, columns=["Time", "Heel", "Toe", "Hip", "Knee", "Ankle"]
    )
    return dataframe


def normalize(array):
    array = array.ravel()
    max = np.max(array)
    min = np.min(array)
    out = (array - min) / (max - min)
    return out


def apply_df_norm(df, column_name):
    array = df[column_name].to_numpy()
    norm_array = normalize(array)
    return norm_array


def lpf(array, order=4, fc=2):
    array = array.ravel()
    # Get filter coefficient
    b, a = sn.iirfilter(N=order, Wn=fc, fs=1000, btype="lowpass", ftype="butter")
    # Forward backward filtering
    out = sn.filtfilt(b, a, array)
    return out


def apply_df_lpf(df, column_name, order, fc):
    array = df[column_name].to_numpy()
    filtered_array = lpf(array, order, fc)
    return filtered_array


def thresholding(array, threshold, width=0.01):
    half_width = width / 2
    # Get the index of points within the range of thershold +- width
    index = np.where(
        np.logical_and(array > threshold + half_width, array < threshold - half_width)
    )

    # Creating a representtation of place in time
    zeros_array = np.zeros_like(array)
    for i in index[0]:
        zeros_array[i] = array[i]
    return zeros_array
