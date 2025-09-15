import base64
import io
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_chart_style(theme: str = "light") -> None:
    """
    Set matplotlib style based on theme.

    Args:
        theme (str): Chart theme, either 'light' or 'dark'
    """
    plt.style.use("seaborn")

    if theme == "dark":
        plt.style.use("dark_background")
        plt.rcParams["axes.facecolor"] = "#1E1E1E"
        plt.rcParams["figure.facecolor"] = "#121212"
        text_color = "white"
    else:
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.facecolor"] = "white"
        text_color = "black"

    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelcolor"] = text_color
    plt.rcParams["xtick.color"] = text_color
    plt.rcParams["ytick.color"] = text_color
    plt.rcParams["text.color"] = text_color


def image_to_base64(fig: Figure, dpi: int = 100, max_width: int = 800) -> str:
    """
    Convert matplotlib figure to base64 encoded PNG.

    Args:
        fig (Figure): Matplotlib figure to convert
        dpi (int): Dots per inch for resolution
        max_width (int): Maximum width in pixels

    Returns:
        str: Base64 encoded image
    """
    try:
        # Adjust figure size to maintain aspect ratio
        width, height = fig.get_size_inches()
        aspect_ratio = height / width

        # Resize if wider than max_width
        if width * dpi > max_width:
            width = max_width / dpi
            height = width * aspect_ratio
            fig.set_size_inches(width, height)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig)
        return base64_image
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return ""


def generate_equity_curve(
    returns: pd.Series,
    drawdown: pd.Series | None = None,
    title: str = "Equity Curve",
    theme: str = "light",
) -> str:
    """
    Generate equity curve with optional drawdown subplot.

    Args:
        returns (pd.Series): Cumulative returns series
        drawdown (pd.Series, optional): Drawdown series
        title (str): Chart title
        theme (str): Chart theme

    Returns:
        str: Base64 encoded image
    """
    set_chart_style(theme)

    try:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}
        )

        # Equity curve
        returns.plot(ax=ax1, linewidth=2, color="blue")
        ax1.set_title(title)
        ax1.set_xlabel("")
        ax1.set_ylabel("Cumulative Returns")
        ax1.grid(True, linestyle="--", alpha=0.7)

        # Drawdown subplot
        if drawdown is not None:
            drawdown.plot(ax=ax2, linewidth=2, color="red")
            ax2.set_title("Maximum Drawdown")
            ax2.set_ylabel("Drawdown (%)")
            ax2.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        return image_to_base64(fig)
    except Exception as e:
        logger.error(f"Error generating equity curve: {e}")
        return ""


def generate_trade_scatter(
    prices: pd.Series,
    trades: pd.DataFrame,
    title: str = "Trade Scatter Plot",
    theme: str = "light",
) -> str:
    """
    Generate trade scatter plot on price chart.

    Args:
        prices (pd.Series): Price series
        trades (pd.DataFrame): Trades DataFrame with entry/exit points
        title (str): Chart title
        theme (str): Chart theme

    Returns:
        str: Base64 encoded image
    """
    set_chart_style(theme)

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot price
        prices.plot(ax=ax, linewidth=1, label="Price", color="blue")

        # Plot entry/exit points
        entry_trades = trades[trades["type"] == "entry"]
        exit_trades = trades[trades["type"] == "exit"]

        ax.scatter(
            entry_trades.index,
            entry_trades["price"],
            color="green",
            marker="^",
            label="Entry",
            s=100,
        )
        ax.scatter(
            exit_trades.index,
            exit_trades["price"],
            color="red",
            marker="v",
            label="Exit",
            s=100,
        )

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        return image_to_base64(fig)
    except Exception as e:
        logger.error(f"Error generating trade scatter plot: {e}")
        return ""


def generate_optimization_heatmap(
    param_results: dict[str, dict[str, float]],
    title: str = "Parameter Optimization",
    theme: str = "light",
) -> str:
    """
    Generate heatmap for parameter optimization results.

    Args:
        param_results (Dict): Dictionary of parameter combinations and performance
        title (str): Chart title
        theme (str): Chart theme

    Returns:
        str: Base64 encoded image
    """
    set_chart_style(theme)

    try:
        # Prepare data for heatmap
        params = list(param_results.keys())
        results = [list(result.values()) for result in param_results.values()]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Custom colormap
        cmap = LinearSegmentedColormap.from_list(
            "performance", ["red", "yellow", "green"]
        )

        sns.heatmap(
            results,
            annot=True,
            cmap=cmap,
            xticklabels=params,
            yticklabels=params,
            ax=ax,
            fmt=".2f",
        )

        ax.set_title(title)
        plt.tight_layout()
        return image_to_base64(fig)
    except Exception as e:
        logger.error(f"Error generating optimization heatmap: {e}")
        return ""


def generate_portfolio_allocation(
    allocations: dict[str, float],
    title: str = "Portfolio Allocation",
    theme: str = "light",
) -> str:
    """
    Generate portfolio allocation pie chart.

    Args:
        allocations (Dict): Dictionary of symbol allocations
        title (str): Chart title
        theme (str): Chart theme

    Returns:
        str: Base64 encoded image
    """
    set_chart_style(theme)

    try:
        fig, ax = plt.subplots(figsize=(8, 8))

        symbols = list(allocations.keys())
        weights = list(allocations.values())

        # Color palette
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(symbols)))

        ax.pie(
            weights,
            labels=symbols,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.85,
        )
        ax.set_title(title)

        plt.tight_layout()
        return image_to_base64(fig)
    except Exception as e:
        logger.error(f"Error generating portfolio allocation chart: {e}")
        return ""


def generate_strategy_comparison(
    strategies: dict[str, pd.Series],
    title: str = "Strategy Comparison",
    theme: str = "light",
) -> str:
    """
    Generate strategy comparison chart.

    Args:
        strategies (Dict): Dictionary of strategy returns
        title (str): Chart title
        theme (str): Chart theme

    Returns:
        str: Base64 encoded image
    """
    set_chart_style(theme)

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        for name, returns in strategies.items():
            returns.plot(ax=ax, label=name, linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        return image_to_base64(fig)
    except Exception as e:
        logger.error(f"Error generating strategy comparison chart: {e}")
        return ""


def generate_performance_dashboard(
    metrics: dict[str, float | str],
    title: str = "Performance Dashboard",
    theme: str = "light",
) -> str:
    """
    Generate performance metrics dashboard as a table image.

    Args:
        metrics (Dict): Dictionary of performance metrics
        title (str): Dashboard title
        theme (str): Chart theme

    Returns:
        str: Base64 encoded image
    """
    set_chart_style(theme)

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")

        # Prepare table data
        metric_names = list(metrics.keys())
        metric_values = [str(val) for val in metrics.values()]

        table = ax.table(
            cellText=[metric_names, metric_values], loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        ax.set_title(title)
        plt.tight_layout()
        return image_to_base64(fig)
    except Exception as e:
        logger.error(f"Error generating performance dashboard: {e}")
        return ""
