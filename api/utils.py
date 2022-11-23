import altair as alt
import plotly.graph_objects as go


def df_line(df, column_name, color="blue"):

    alt_chart = (
        alt.Chart(df)
        .mark_line(color=color)
        .encode(x="Time", y=column_name, tooltip=["Time", column_name])
        .interactive()
    )
    return alt_chart


def add_one_line(fig, df, x, y, name):
    fig.add_trace(
        go.Scattergl(
            x=df[x],
            y=df[y],
            name=name,
            mode="lines",
        )
    )


def add_one_scatter(fig, df, x, y, name):
    fig.add_trace(
        go.Scattergl(
            x=df[x],
            y=df[y],
            name=name,
            mode="markers",
        )
    )


def add_line(fig, df, x, y, names):

    if len(y) != len(names):
        raise Exception("Length y != Length name")

    for col, name in zip(y, names):
        add_one_line(fig, df, x, col, name)


def add_scatter(fig, df, x, y, names):

    if len(y) != len(names):
        raise Exception("Length y != Length name")

    for col, name in zip(y, names):
        add_one_scatter(fig, df, x, col, name)
