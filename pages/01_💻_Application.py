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
        c8, c9 = st.columns(2)

        order = c5.number_input("LPF Filter Order", value=3, min_value=1, max_value=10)
        fc = c5.number_input(
            "LPF Filter Cutoff Frequency",
            value=2.0,
            min_value=0.0,
            max_value=1000.0,
            format="%f",
        )

        c6.markdown("#### 1. Filtering")
        file_df["Filtered Heel"] = logic.apply_df_lpf(
            file_df, "Heel", order=order, fc=fc
        )
        file_df["Filtered Toe"] = logic.apply_df_lpf(file_df, "Toe", order=order, fc=fc)
        c6.line_chart(file_df, x="Time", y=["Filtered Heel", "Filtered Toe"])

        c7.markdown("#### 2. Normalization")
        file_df["Normalized Heel"] = logic.apply_df_norm(file_df, "Filtered Heel")
        file_df["Normalized Toe"] = logic.apply_df_norm(file_df, "Filtered Toe")
        c7.line_chart(file_df, x="Time", y=["Normalized Heel", "Normalized Toe"])


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
