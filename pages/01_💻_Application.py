import altair as alt
import streamlit as st

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

        order = c5.number_input("LPF Filter Order", value=3, min_value=1, max_value=10)
        fc = c5.number_input(
            "LPF Filter Cutoff Frequency",
            value=2.0,
            min_value=0.0,
            max_value=1000.0,
            format="%f",
        )

        c6.markdown("#### Filtering")
        file_df["Filtered Heel"] = logic.apply_df_lpf(
            file_df, "Heel", order=order, fc=fc
        )
        file_df["Filtered Toe"] = logic.apply_df_lpf(file_df, "Toe", order=order, fc=fc)
        c6.line_chart(file_df, x="Time", y=["Filtered Heel", "Filtered Toe"])

        c7.markdown("#### Normalization")
        file_df["Normalized Heel"] = logic.apply_df_norm(file_df, "Filtered Heel")
        file_df["Normalized Toe"] = logic.apply_df_norm(file_df, "Filtered Toe")
        c7.line_chart(file_df, x="Time", y=["Normalized Heel", "Normalized Toe"])

        threshold = c8.number_input(
            "Gait Threshold", value=0.05, min_value=0.0, max_value=1.0, format="%f"
        )
        width = c8.number_input(
            "Threshold Search Width",
            value=0.01,
            min_value=0.0,
            max_value=1.0,
            format="%f",
        )
        c9.markdown("#### Thresholding")
        th_heel = logic.apply_df_thresholding(
            file_df, "Normalized Heel", threshold, width
        )
        th_toe = logic.apply_df_thresholding(
            file_df, "Normalized Toe", threshold, width
        )

        c_norm_heel = utils.df_line(file_df, "Normalized Heel")
        c_norm_toe = utils.df_line(file_df, "Normalized Toe")
        c_th_heel = (
            alt.Chart(th_heel)
            .mark_circle(color="red")
            .encode(x="Time", y="Threshold", tooltip=["Time", "Threshold"])
            .interactive()
        )
        c_th_toe = (
            alt.Chart(th_toe)
            .mark_circle(color="red")
            .encode(x="Time", y="Threshold", tooltip=["Time", "Threshold"])
            .interactive()
        )
        layer = alt.layer(c_norm_heel, c_norm_toe, c_th_toe, c_th_heel)

        c9.altair_chart(layer, use_container_width=True)
        # c9.line_chart(file_df, x="Time", y=["Thresholded Heel", "Normalized Heel"])


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
