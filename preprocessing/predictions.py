import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
from numpy.typing import NDArray
from myfeatures.dates import financial_date_correction


def format_predictions(
    predictions: NDArray[np.float64], features: pd.DataFrame
) -> pd.DataFrame:
    """Format model predictions into a dataframe thats easier to process

    Args:
        predictions (NDArray[np.float64]): predictions directly from the tensorflow model
        features (pd.DataFrame): machine learninf model features used to get date indices for predictions

    Returns:
        pd.DataFrame: formatted predictions
    """
    predictions = np.squeeze(predictions)  # remove additional dimension added from tf

    df = pd.DataFrame([])
    n_cols = predictions.shape[1]
    for idx in range(n_cols):
        pred_col = predictions[:, idx]
        time_idx = pd.DatetimeIndex(
            [dt + relativedelta(days=idx + 1) for dt in features.index]
        )
        corrected_time_idx = [
            financial_date_correction(t, direction="forward") for t in time_idx
        ]
        df_append = pd.DataFrame(
            {f"label_{idx + 1}": pred_col}, index=corrected_time_idx
        )
        df = df.join(df_append) if not df.empty else df_append

    return df


def performance_plot(predictions: pd.DataFrame, financial_data: pd.DataFrame) -> Figure:
    """Creates plot to show how the machine learning model predictions vs the ground truth stock close data over time

    Args:
        predictions (pd.DataFrame): predictions from machine learning model
        financial_data (pd.DataFrame): ground truth stock data within the time frame of the predictions

    Returns:
        Figure: matplotlib figure showing the predictions vs ground truth stock data
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(
        financial_data.index, financial_data["Close"], linewidth=4, label="Ground Truth"
    )
    for col_name, col in predictions.items():
        ax.scatter(col.index, col, s=3, label=col_name)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", labelrotation=45)

    ax.grid()
    ax.legend()
    ax.set_title("Ground Truth Close Price vs Predictions", fontsize=16)
    ax.set_xlabel("Time (days)", fontsize=14)
    ax.set_ylabel("Stock Price ($)", fontsize=14)

    return fig
