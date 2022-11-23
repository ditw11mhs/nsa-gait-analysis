import altair as alt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from api import logic, utils


def main():
    st.header("Gait Analysis")
    file = st.file_uploader("Upload data here")
    if file is not None:
        file_df = logic.txt_to_df(file)

        st.subheader("Raw Data")
        # Raw Data visualization

        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        c1.line_chart(file_df, x="Time", y=["Heel", "Toe"])
        c2.line_chart(file_df, x="Time", y=["Hip"])
        c3.line_chart(file_df, x="Time", y=["Knee"])
        c4.line_chart(file_df, x="Time", y=["Ankle"])

        st.subheader("Gait Analysis")
        c5, c6, c7 = st.columns([1, 2, 2])
        c8, c9 = st.columns([1, 2])
        c10, c11 = st.columns(2)

        order = c5.number_input(
            "LPF Filter Order", value=3, min_value=1, max_value=10)
        fc = c5.number_input(
            "LPF Filter Cutoff Frequency",
            value=4.0,
            min_value=0.0,
            max_value=1000.0,
            format="%f",
        )

        c6.markdown("#### Filtering")
        file_df["Filtered Heel"] = logic.apply_df_lpf(
            file_df, "Heel", order=order, fc=fc, zero_thres=True
        )
        file_df["Filtered Toe"] = logic.apply_df_lpf(
            file_df, "Toe", order=order, fc=fc, zero_thres=True
        )

        file_df["Filtered Knee"] = logic.apply_df_lpf(
            file_df, "Knee", order=order, fc=fc
        )

        file_df["Filtered Hip"] = logic.apply_df_lpf(
            file_df, "Hip", order=order, fc=fc)

        file_df["Filtered Ankle"] = logic.apply_df_lpf(
            file_df, "Ankle", order=order, fc=fc
        )

        c6.line_chart(file_df, x="Time", y=["Filtered Heel", "Filtered Toe"])

        c7.markdown("#### Normalization")

        file_df["Normalized Heel"] = logic.apply_df_norm(
            file_df, "Filtered Heel")
        file_df["Normalized Toe"] = logic.apply_df_norm(
            file_df, "Filtered Toe")

        c7.line_chart(file_df, x="Time", y=[
                      "Normalized Heel", "Normalized Toe"])

        threshold = c8.number_input(
            "Gait Threshold", value=0.05, min_value=0.0, max_value=1.0, format="%f"
        )

        c9.markdown("#### Thresholding")
        th_heel, index_heel = logic.apply_df_thresholding(
            file_df, "Normalized Heel", threshold
        )
        th_toe, index_toe = logic.apply_df_thresholding(
            file_df, "Normalized Toe", threshold
        )

        threshold_fig = go.Figure()
        utils.add_line(
            threshold_fig,
            file_df,
            "Time",
            ["Normalized Toe", "Normalized Heel"],
            ["Normalized Toe", "Normalized Heel"],
        )
        for th, name in zip([th_heel, th_toe], ["Heel Threshold", "Toe Threshold"]):
            utils.add_one_scatter(threshold_fig, th, "Time", "Threshold", name)

        c9.plotly_chart(threshold_fig, use_container_width=True)

        cycle = st.number_input("Cycle", min_value=1)
        gait_df, gait_param_df = logic.split_gait_cycle(
            file_df["Normalized Heel"],
            file_df["Normalized Toe"],
            index_heel,
            index_toe,
            cycle,
        )

        gait_fig = go.Figure()
        utils.add_line(
            gait_fig,
            gait_df,
            "Time",
            ["Gait Heel", "Gait Toe"],
            ["Gait Heel", "Gait Toe"],
        )
        gait_fig.add_traces(
            list(
                px.scatter(
                    gait_param_df, x="Index", y="Value", color="Param"
                ).select_traces()
            )
        )

        st.plotly_chart(gait_fig, use_container_width=True)
        (
            cycle_joint_df,
            knee_param,
            ankle_param,
            hip_param,
        ) = logic.split_gait_joints_cyle(
            file_df, gait_df, gait_param_df, index_heel, index_toe, cycle
        )

        knee_fig = go.Figure()
        utils.add_one_line(knee_fig, cycle_joint_df,
                           "Time", "Gait Knee", "Gait Knee")
        knee_fig.add_traces(
            list(
                px.scatter(
                    knee_param, x="Index", y="Value", color="Param"
                ).select_traces()
            )
        )
        st.plotly_chart(knee_fig, use_container_width=True)

        ankle_fig = go.Figure()
        utils.add_one_line(
            ankle_fig, cycle_joint_df, "Time", "Gait Ankle", "Gait Ankle"
        )
        ankle_fig.add_traces(
            list(
                px.scatter(
                    ankle_param, x="Index", y="Value", color="Param"
                ).select_traces()
            )
        )
        st.plotly_chart(ankle_fig, use_container_width=True)

        hip_fig = go.Figure()
        utils.add_one_line(hip_fig, cycle_joint_df,
                           "Time", "Gait Hip", "Gait Hip")
        hip_fig.add_traces(
            list(
                px.scatter(
                    hip_param, x="Index", y="Value", color="Param"
                ).select_traces()
            )
        )
        st.plotly_chart(hip_fig, use_container_width=True)

        ####### MUSCLE ACTIVATION#####
        ##############################

        st.subheader("Muscle Activation")
        c1m, c2m, c3m = st.columns(3)

        # Filter
        muscle_col = [
            "Gluteus Maximus",
            "Bicep Femoris Short",
            "Bicep Femoris Long",
            "Vastus Medialis",
            "Vastus Lateralis",
            "Rectus Remoris",
            "Soleus",
            "Gastronecmius",
            "Tibialis Anterior",
        ]

        for muscle, col in zip(
            muscle_col, [c1m, c2m, c3m, c1m, c2m, c3m, c1m, c2m, c3m]
        ):
            col.line_chart(file_df, x="Time", y=muscle)

        st.markdown("#### Rectification and Filtering")
        mav_window = st.number_input("MAV window", value=150, min_value=1)

        for muscle in muscle_col:
            file_df[f"Rectified {muscle}"] = logic.apply_df_rect(
                file_df, muscle)
            file_df[f"Filtered {muscle}"] = logic.apply_df_mav(
                file_df, f"Rectified {muscle}", mav_window
            )
        c4m, c5m, c6m = st.columns(3)

        st.markdown("#### Cycle and Thresholding")
        muscle_th = st.number_input(
            "Muscle Thresholding", value=0.7, min_value=0.0, max_value=1.0
        )
        for muscle, col in zip(
            muscle_col, [c4m, c5m, c6m, c4m, c5m, c6m, c4m, c5m, c6m]
        ):
            col.line_chart(file_df, x="Time", y=f"Filtered {muscle}")

        c7m, c8m, c9m = st.columns(3)
        ic, ho, ff, to = logic.gait_param(index_heel, index_toe)

        for muscle, col in zip(
            muscle_col, [c7m, c8m, c9m, c7m, c8m, c9m, c7m, c8m, c9m]
        ):
            muscle_act_cycle = logic.normalize(
                logic.split_gait(file_df[f"Filtered {muscle}"], ic, cycle)
            )
            muscle_act_cycle_time = logic.split_gait(
                file_df["Time"], ic, cycle)

            muscle_thresholding = logic.muscle_thresholding(
                muscle_act_cycle, muscle_th)

            muscle_df = pd.DataFrame(
                {
                    "Time": muscle_act_cycle_time,
                    f"Filtered {muscle} Cycle {cycle}": muscle_act_cycle,
                    f"Thresholding {muscle}": muscle_thresholding,
                }
            )
            col.line_chart(
                muscle_df,
                x="Time",
                y=[f"Filtered {muscle} Cycle {cycle}",
                    f"Thresholding {muscle}"],
            )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
