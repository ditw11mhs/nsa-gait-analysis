import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal as sn


@st.cache(allow_output_mutation=True)
def txt_to_df(file):
    data_array = np.loadtxt(file)
    dataframe = pd.DataFrame(
        data_array,
        columns=[
            "Time",
            "Heel",
            "Toe",
            "Hip",
            "Knee",
            "Ankle",
            "Gluteus Maximus",
            "Bicep Femoris Short",
            "Bicep Femoris Long",
            "Vastus Medialis",
            "Vastus Lateralis",
            "Rectus Remoris",
            "Soleus",
            "Gastronecmius",
            "Tibialis Anterior",
        ],
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

    # TODO: Implement filter (order 2)
    # Notes: EMG 0.5-1 Hz
    # Online implementation need optimization to make things work

    # Get filter coefficient
    b, a = sn.iirfilter(N=order, Wn=fc, fs=1000,
                        btype="lowpass", ftype="butter")

    # Forward backward filtering
    out = sn.filtfilt(b, a, array)

    # Remove negative to 0
    out = np.where(out < 0, 0, out)

    return out


def apply_df_lpf(df, column_name, order, fc):
    array = df[column_name].to_numpy()
    filtered_array = lpf(array, order, fc)
    return filtered_array


def thresholding(array, time_array, threshold):
    # Use falling and rising edge detection with convolve trick
    sign = array > threshold
    convolve = np.convolve(sign, [1, -1])
    index = np.array(np.where(np.logical_or(
        convolve == 1, convolve == -1))) - 1
    print(index)

    # TODO: Add logic to get the first sample that is bigger than threshold

    df = pd.DataFrame(
        {"Time": time_array[index[0]], "Threshold": array[index[0]]})

    return df


def apply_df_thresholding(df, column_name, threshold):
    array = df[column_name].to_numpy()
    time_array = df["Time"].to_numpy()
    thresholded_array = thresholding(array, time_array, threshold)

    return thresholded_array
