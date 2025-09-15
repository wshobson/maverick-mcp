"""
Visualization utilities for Maverick-MCP.

This module contains functions for generating charts and visualizations
for financial data, including technical analysis charts.
"""

import base64
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

from maverick_mcp.config.plotly_config import setup_plotly

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("maverick_mcp.visualization")

# Configure Plotly to use modern defaults and suppress warnings
setup_plotly()


def plotly_fig_to_base64(fig: go.Figure, format: str = "png") -> str:
    """
    Convert a Plotly figure to a base64 encoded data URI string.

    Args:
        fig: The Plotly figure to convert
        format: Image format (default: 'png')

    Returns:
        Base64 encoded data URI string of the figure
    """
    img_bytes = None
    with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as tmpfile:
        try:
            fig.write_image(tmpfile.name)
            tmpfile.seek(0)
            img_bytes = tmpfile.read()
        except Exception as e:
            logger.error(f"Error writing image: {e}")
            raise
    os.remove(tmpfile.name)
    if not img_bytes:
        logger.error("No image bytes were written. Is kaleido installed?")
        raise RuntimeError(
            "Plotly failed to write image. Ensure 'kaleido' is installed."
        )
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{format};base64,{base64_str}"


def create_plotly_technical_chart(
    df: pd.DataFrame, ticker: str, height: int = 400, width: int = 600
) -> go.Figure:
    """
    Generate a Plotly technical analysis chart for financial data visualization.

    Args:
        df: DataFrame with price and technical indicator data
        ticker: The ticker symbol to display in the chart title
        height: Chart height
        width: Chart width

    Returns:
        A Plotly figure with the technical analysis chart
    """
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]
    df = df.iloc[-126:].copy()  # Ensure we keep DataFrame structure

    fig = sp.make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("", "", "", ""),
        row_heights=[0.6, 0.15, 0.15, 0.1],
    )

    bg_color = "#FFFFFF"
    text_color = "#000000"
    grid_color = "rgba(0, 0, 0, 0.35)"
    colors = {
        "green": "#00796B",
        "red": "#D32F2F",
        "blue": "#1565C0",
        "orange": "#E65100",
        "purple": "#6A1B9A",
        "gray": "#424242",
        "black": "#000000",
    }
    line_width = 1

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            name="Price",
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=colors["green"],
            decreasing_line_color=colors["red"],
            line={"width": line_width},
        ),
        row=1,
        col=1,
    )

    # Moving averages
    for i, (col, name) in enumerate(
        [("ema_21", "EMA 21"), ("sma_50", "SMA 50"), ("sma_200", "SMA 200")]
    ):
        color = [colors["blue"], colors["green"], colors["red"]][i]
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=name,
                line={"color": color, "width": line_width},
            ),
            row=1,
            col=1,
        )

    # Bollinger Bands
    light_blue = "rgba(21, 101, 192, 0.6)"
    fill_color = "rgba(21, 101, 192, 0.1)"
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["bbu_20_2.0"],
            mode="lines",
            line={"color": light_blue, "width": line_width},
            name="Upper BB",
            legendgroup="bollinger",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["bbl_20_2.0"],
            mode="lines",
            line={"color": light_blue, "width": line_width},
            name="Lower BB",
            legendgroup="bollinger",
            showlegend=False,
            fill="tonexty",
            fillcolor=fill_color,
        ),
        row=1,
        col=1,
    )

    # Volume
    volume_colors = np.where(df["close"] >= df["open"], colors["green"], colors["red"])
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker={"color": volume_colors},
            opacity=0.75,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["rsi"],
            mode="lines",
            name="RSI",
            line={"color": colors["blue"], "width": line_width},
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color=colors["red"],
        line_width=line_width,
        row=3,
        col=1,
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color=colors["green"],
        line_width=line_width,
        row=3,
        col=1,
    )

    # MACD
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["macd_12_26_9"],
            mode="lines",
            name="MACD",
            line={"color": colors["blue"], "width": line_width},
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["macds_12_26_9"],
            mode="lines",
            name="Signal",
            line={"color": colors["orange"], "width": line_width},
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["macdh_12_26_9"],
            name="Histogram",
            showlegend=False,
            marker={"color": df["macdh_12_26_9"], "colorscale": "RdYlGn"},
        ),
        row=4,
        col=1,
    )

    # Layout
    import datetime

    now = datetime.datetime.now(datetime.UTC).strftime("%m/%d/%Y")
    fig.update_layout(
        height=height,
        width=width,
        title={
            "text": f"<b>{ticker.upper()} | {now} | Technical Analysis | Maverick-MCP</b>",
            "font": {"size": 12, "color": text_color, "family": "Arial, sans-serif"},
            "y": 0.98,
        },
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        xaxis_rangeslider_visible=False,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 10, "color": text_color, "family": "Arial, sans-serif"},
            "itemwidth": 30,
            "itemsizing": "constant",
            "borderwidth": 0,
            "tracegroupgap": 1,
        },
        font={"size": 10, "color": text_color, "family": "Arial, sans-serif"},
        margin={"r": 20, "l": 40, "t": 80, "b": 0},
    )

    fig.update_xaxes(
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        zerolinewidth=line_width,
        gridwidth=1,
        griddash="dot",
    )
    fig.update_yaxes(
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        zerolinewidth=line_width,
        gridwidth=1,
        griddash="dot",
    )

    y_axis_titles = ["Price", "Volume", "RSI", "MACD"]
    for i, title in enumerate(y_axis_titles, start=1):
        if title:
            fig.update_yaxes(
                title={
                    "text": f"<b>{title}</b>",
                    "font": {"size": 8, "color": text_color},
                    "standoff": 0,
                },
                side="left",
                position=0,
                automargin=True,
                row=i,
                col=1,
                tickfont={"size": 8},
            )

    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=3, col=1)
    fig.update_xaxes(
        title={"text": "Date", "font": {"size": 8, "color": text_color}, "standoff": 5},
        row=4,
        col=1,
        tickfont={"size": 8},
        showticklabels=True,
        tickangle=45,
        tickformat="%Y-%m-%d",
    )

    return fig
