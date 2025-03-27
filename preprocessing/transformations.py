import pandas as pd
import tensorflow as tf


def _add_dimension_to_element(
    feature: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    feature_expanded = tf.expand_dims(feature, axis=0)
    label_expanded = tf.expand_dims(label, axis=0)
    return feature_expanded, label_expanded


def df_to_dataset(
    df: pd.DataFrame,
    batch_size: int,
    label_pattern: str = "label",
) -> tf.data.Dataset:
    _df = df.copy()
    feature_cols = [col for col in _df.columns if label_pattern not in col]
    features = _df[feature_cols]
    labels = _df.filter(like=label_pattern)

    return (
        tf.data.Dataset.from_tensor_slices((features, labels))
        .map(_add_dimension_to_element)
        # .shuffle(buffer_size=_df.shape[0])
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
