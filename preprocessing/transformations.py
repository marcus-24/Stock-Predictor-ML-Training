import pandas as pd
import tensorflow as tf


def _add_dimension_to_element(
    feature: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Add a batch dimension to features and label tensors

    Args:
        feature (tf.Tensor): feature tensor
        label (tf.Tensor): label tensor

    Returns:
        tuple[tf.Tensor, tf.Tensor]: feature and label tensor with the extra batch dimensions
    """
    feature_expanded = tf.expand_dims(feature, axis=0)
    label_expanded = tf.expand_dims(label, axis=0)
    return feature_expanded, label_expanded


def df_to_dataset(
    df: pd.DataFrame,
    batch_size: int,
    label_pattern: str = "label",
) -> tf.data.Dataset:
    """Transform a dataframe containing features and labels to a tensorflow dataset

    Args:
        df (pd.DataFrame): contains features and label data.
        batch_size (int): the number of samples for each batch.
        label_pattern (str, optional): the consistent string pattern within all label
        column names. Defaults to "label".

    Returns:
        tf.data.Dataset: tensorflow dataset with feature and label batches
    """
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
