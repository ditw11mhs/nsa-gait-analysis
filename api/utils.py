import altair as alt


def df_line(df, column_name):

    alt_chart = (
        alt.Chart(df)
        .mark_line()
        .encode(x="Time", y=column_name, tooltip=["Time", column_name])
        .interactive()
    )
    return alt_chart
