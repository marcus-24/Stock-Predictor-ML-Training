import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import pandas as pd
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from numpy.typing import NDArray
import holidays

HOLIDAYS = holidays.financial_holidays("NYSE")


def financial_time_correction(
    original_time: date, my_holidays: dict[datetime, str] = HOLIDAYS
) -> date:
    corrected_time = original_time
    last_weekday_num = 4  # any weekday number greater is a weekend
    while corrected_time in my_holidays or corrected_time.weekday() > last_weekday_num:
        corrected_time += relativedelta(days=1)

    return corrected_time


def format_predictions(
    predictions: NDArray[np.float64], features: pd.DataFrame
) -> pd.DataFrame:
    predictions = np.squeeze(predictions)  # remove additional dimension added from tf

    df = pd.DataFrame([])
    n_cols = predictions.shape[1]
    for idx in range(n_cols):
        pred_col = predictions[:, idx]
        time_idx = pd.DatetimeIndex(
            [dt + relativedelta(days=idx + 1) for dt in features.index]
        )
        corrected_time_idx = [financial_time_correction(t) for t in time_idx]
        df_append = pd.DataFrame(
            {f"label_{idx + 1}": pred_col}, index=corrected_time_idx
        )
        df = df.join(df_append) if not df.empty else df_append

    return df


def custom_plot(predictions: pd.DataFrame, financial_data: pd.DataFrame) -> Figure:
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

    # plt.show()

    return fig
