import pandas as pd
import tensorflow as tf
from scipy import stats


def feature_engineering(df: pd.DataFrame, win_size: int, n_future: int) -> pd.DataFrame:
    # https://medium.com/aimonks/improving-stock-price-forecasting-by-feature-engineering-8a5d0be2be96
    _df = df.copy()

    trans_df = pd.DataFrame(
        {
            "daily_var": (_df["High"] - _df["Low"]) / (_df["Open"]),
            "sev_day_sma": _df["Close"].rolling(win_size).mean(),
            "sev_day_std": _df["Close"].rolling(win_size).std(),
            "daily_return": _df["Close"].diff(),
            "sma_2std_pos": _df["Close"].rolling(win_size).mean()
            + 2 * _df["Close"].rolling(win_size).std(),
            "sma_2std_neg": _df["Close"].rolling(win_size).mean()
            - 2 * _df["Close"].rolling(win_size).std(),
            "high_close": (_df["High"] - _df["Close"]) / _df["Open"],
            "low_open": (_df["Low"] - _df["Open"]) / _df["Open"],
            "cumul_return": _df["Close"] - _df["Close"].iloc[0],
            "label": _df["Close"].shift(n_future),
        }
    ).dropna()
    return trans_df


def feature_selection(df: pd.DataFrame, label_col_name: str = "label"):
    _df = df.copy()
    label = _df.pop(label_col_name)
    return [(col_name, *stats.ttest_ind(label, ser)) for col_name, ser in _df.items()]


def add_dimension_to_element(
    feature: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    feature_expanded = tf.expand_dims(feature, axis=0)
    label_expanded = tf.expand_dims(label, axis=0)
    return feature_expanded, label_expanded


def df_to_dataset(
    df: pd.DataFrame,
    batch_size: int,
    label_col: str = "label",
) -> tf.data.Dataset:
    _df = df.copy()
    labels = _df.pop(label_col)

    return (
        tf.data.Dataset.from_tensor_slices((_df, labels))
        .map(add_dimension_to_element)
        # .shuffle(buffer_size=_df.shape[0])
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
