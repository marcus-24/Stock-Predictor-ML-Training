import pandas as pd
import tensorflow as tf


def windowed_mean(x: tf.Tensor) -> tf.Tensor:
    mean = tf.math.reduce_mean(x, axis=0)
    return tf.expand_dims(mean, axis=0)


def sequential_window_dataset(
    df: pd.DataFrame,
    batch_size: int,
    n_past: int,
    n_future: int,
    shift: int,
    shuffle_batch_size: int = 10000,
) -> tf.data.Dataset:

    return (
        tf.data.Dataset.from_tensor_slices(
            df.values
        )  # transform array to tensor dataset type
        .window(
            size=n_past + n_future, shift=shift, drop_remainder=True
        )  # window features
        .flat_map(
            lambda window: window.batch(n_past + n_future)
        )  # convert to tensors of given batch size
        .shuffle(shuffle_batch_size)  # shuffle data to avoid learning patterns
        .map(
            lambda window: (
                windowed_mean(window[:n_past]),
                window[-1, -1],
            )  # get last point as prediction
        )  # take one window at a time and split into features and labels
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )  # create batch and prefetch next batch while processing current batch
