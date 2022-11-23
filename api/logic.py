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


def apply_df_rect(df, column_name):
    array = df[column_name].to_numpy()
    rect_array = np.absolute(array)
    return rect_array


def apply_df_mav(df, column_name, window):
    array = df[column_name].to_numpy()
    mav_array = moving_average(array, window)
    return mav_array


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

    return out


def apply_df_lpf(df, column_name, order, fc, zero_thres=False):
    array = df[column_name].to_numpy()
    filtered_array = lpf(array, order, fc)
    # Remove negative to 0
    if zero_thres:
        filtered_array = np.where(filtered_array < 0, 0, filtered_array)
    return filtered_array


def thresholding(array, time_array, threshold):
    # Use falling and rising edge detection with convolve trick
    sign = array > threshold
    convolve = np.convolve(sign, [1, -1])
    index = np.array(np.where(np.logical_or(
        convolve == 1, convolve == -1))) - 1

    df = pd.DataFrame(
        {"Time": time_array[index[0]], "Threshold": array[index[0]]})

    return df, index[0]


def apply_df_thresholding(df, column_name, threshold):
    array = df[column_name].to_numpy()
    time_array = df["Time"].to_numpy()
    df_thresholded, index = thresholding(array, time_array, threshold)

    return df_thresholded, index


def gait_param_heel(index_th_heel):
    ic = index_th_heel[::2]
    ho = index_th_heel[1::2]
    return ic, ho


def gait_param_toe(index_th_toe):
    ff = index_th_toe[::2]
    to = index_th_toe[1::2]
    return ff, to


def gait_param(index_th_heel, index_th_toe):
    ic, ho = gait_param_heel(index_th_heel)

    ff, to = gait_param_toe(index_th_toe)
    return ic, ho, ff, to


def gait_param_cycle(index_th_heel, index_th_toe, cycle):
    ic, ho, ff, to = gait_param(index_th_heel, index_th_toe)

    ic_first = ic[cycle - 1]
    ic_cycle = [0, ic[cycle] - ic_first]
    ho_cycle = ho[cycle - 1] - ic_first
    ff_cycle = ff[cycle - 1] - ic_first
    to_cycle = to[cycle - 1] - ic_first
    return ic_cycle, ho_cycle, ff_cycle, to_cycle


def scaler(array, new_min, new_max):
    array = array.ravel()
    max = np.max(array)
    min = np.min(array)
    out = ((array - min) * (new_max - new_min) / (max - min)) + new_min
    return out


def split_gait(array, ic, cycle):
    # cycle start from 1, and index start from 0
    start_gait = ic[cycle - 1]
    end_gait = ic[cycle] + 1
    return array[start_gait:end_gait]


def split_gait_cycle(array_heel, array_toe, index_th_heel, index_th_toe, cycle):
    ic, ho, ff, to = gait_param(index_th_heel, index_th_toe)
    ic_cycle, ho_cycle, ff_cycle, to_cycle = gait_param_cycle(
        index_th_heel, index_th_toe, cycle
    )

    gait_heel = split_gait(array_heel, ic, cycle)
    gait_toe = split_gait(array_toe, ic, cycle)

    time_array = np.linspace(0, 100, len(gait_heel))
    index = np.array([ic_cycle[0], ho_cycle, ff_cycle, to_cycle, ic_cycle[1]])
    norm_index = time_array[index]

    df = pd.DataFrame(
        {"Time": time_array, "Gait Heel": gait_heel, "Gait Toe": gait_toe}
    )
    param = pd.DataFrame(
        {
            "Param": ["IC", "HO", "FF", "TO", "IC"],
            "Index": norm_index,
            "Value": [
                array_heel[ic[cycle - 1]],
                array_heel[ho[cycle - 1]],
                array_toe[ff[cycle - 1]],
                array_toe[to[cycle - 1]],
                array_heel[ic[cycle]],
            ],
        }
    )
    return df, param


def split_gait_joints_cyle(
    file_df, gait_df, gait_param_df, index_th_heel, index_th_toe, cycle
):
    ic, ho, ff, to = gait_param(index_th_heel, index_th_toe)
    ic_cycle, ho_cycle, ff_cycle, to_cycle = gait_param_cycle(
        index_th_heel, index_th_toe, cycle
    )

    array_knee = file_df["Filtered Knee"].to_numpy()
    array_ankle = file_df["Filtered Ankle"].to_numpy()
    array_hip = file_df["Filtered Hip"].to_numpy()

    cycle_knee = split_gait(array_knee, ic, cycle)
    cycle_ankle = split_gait(array_ankle, ic, cycle)
    cycle_hip = split_gait(array_hip, ic, cycle)

    min_max_knee, index_min_max_knee = min_max_stance_swing(
        cycle_knee, to_cycle)
    min_max_ankle, index_min_max_ankle = min_max_stance_swing(
        cycle_ankle, to_cycle)
    min_max_hip, index_min_max_hip = min_max_stance_swing(cycle_hip, to_cycle)

    cycle_joint_df = pd.DataFrame(
        {
            "Time": gait_df["Time"],
            "Gait Knee": cycle_knee,
            "Gait Ankle": cycle_ankle,
            "Gait Hip": cycle_hip,
        }
    )

    knee_param = pd.DataFrame(
        {
            "Param": ["MKEst", "MKFst", "MKEsw", "MKFsw"],
            "Index": gait_df["Time"].to_numpy()[index_min_max_knee],
            "Value": min_max_knee,
        }
    )
    ankle_param = pd.DataFrame(
        {
            "Param": ["MAPFst", "MADFst", "MAPFsw", "MADFsw"],
            "Index": gait_df["Time"].to_numpy()[index_min_max_ankle],
            "Value": min_max_ankle,
        }
    )
    hip_param = pd.DataFrame(
        {
            "Param": ["MHEst", "MHFsw"],
            "Index": gait_df["Time"].to_numpy()[
                [index_min_max_hip[0], index_min_max_hip[3]]
            ],
            "Value": [min_max_hip[0], min_max_hip[3]],
        }
    )

    return cycle_joint_df, knee_param, ankle_param, hip_param


def min_max_stance_swing(array, to):
    array_stance = array[:to]
    array_swing = array[to:]

    min_array_stance = np.min(array_stance)
    max_array_stance = np.max(array_stance)
    min_array_swing = np.min(array_swing)
    max_array_swing = np.max(array_swing)

    index_min_array_stance = np.argmin(array_stance)
    index_max_array_stance = np.argmax(array_stance)
    index_min_array_swing = np.argmin(array_swing) + to
    index_max_array_swing = np.argmax(array_swing) + to

    min_max = [min_array_stance, max_array_stance,
               min_array_swing, max_array_swing]
    index_min_max = [
        index_min_array_stance,
        index_max_array_stance,
        index_min_array_swing,
        index_max_array_swing,
    ]
    return min_max, index_min_max


def moving_average(array, window):
    return np.convolve(array, np.ones(window), "same") / window


def muscle_thresholding(array, threshold):
    out = array >= threshold
    return out * 1


def apply_df_musscle_th(df, column_name, threshold):
    array = df[column_name].to_numpy()
    out = muscle_thresholding(array, threshold)
    return out
