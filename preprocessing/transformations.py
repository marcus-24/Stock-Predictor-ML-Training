import pandas as pd
import tensorflow as tf


def sequential_window_dataset(
    df: pd.DataFrame, batch_size: int, n_past: int, n_future: int, shift: int
) -> tf.data.Dataset:

    return (
        tf.data.Dataset.from_tensor_slices(
            df.values
        )  # transform array to tensor dataset type
        .window(
            size=n_past + n_future, shift=shift, drop_remainder=True
        )  # window features
        .flat_map(lambda w: w.batch(n_past + n_future))  #
        .map(
            lambda w: (w[:n_past], w[n_past:, -1])
        )  # split into features and labels (window past )
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )  # create batch and prefetch next batch while processing current batch
