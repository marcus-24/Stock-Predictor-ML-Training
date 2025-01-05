import pandas as pd
import yfinance as yf
import tensorflow as tf
import keras
from keras import layers, models, optimizers, losses  # TODO: figure out pylance errors
import os
import warnings
import datetime

warnings.filterwarnings("ignore", message="You are saving your model as an HDF5 file.")


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # needed to suppress out of rand warnings

DATA_COLS = ["Open", "High", "Low", "Close"]
BATCH_SIZE = 32
N_PAST = 10
N_FUTURE = 10
N_SHIFT = 1
EPOCHS = 1000
TRAIN_PERCENT = 0.75
N_FEATURES = len(DATA_COLS)


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
            lambda w: (w[:n_past], w[n_past:])
        )  # split into features and labels (window past )
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )  # create batch and prefetch next batch while processing current batch


def build_bidirec_lstm_model(
    data: tf.data.Dataset, n_past: int, n_features: int, batch_size: int
) -> models.Sequential:

    # TODO: Fine tune normalization layer. Performance is lower than scikit-learn standard scaler
    norm_layer = layers.Normalization()
    norm_layer.adapt(
        data.map(lambda x, _: x)
    )  # need to calculate the mean and variance for z-score (map used to extract only features and ignore labels)

    model = models.Sequential(
        [
            layers.InputLayer(shape=(n_past, n_features), batch_size=batch_size),
            norm_layer,  # plug in fitted normalization layer
            layers.Bidirectional(layers.LSTM(5, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(5, return_sequences=True)),
            layers.Dense(n_features),
        ]
    )

    optimizer = optimizers.SGD(learning_rate=0.1, momentum=0.9)
    model.compile(loss=losses.Huber(), optimizer=optimizer, metrics=["mae"])
    model.summary()

    return model


if __name__ == "__main__":

    df = yf.Ticker("AAPL").history(interval="1d", start="2016-01-01", end="2024-01-30")

    """Split Data"""
    split_idx = int(TRAIN_PERCENT * df.shape[0])
    split_time = df.index[split_idx]

    x_train = df.loc[:split_time, DATA_COLS]
    train_time = x_train.index.to_numpy()
    x_val = df.loc[split_time:, DATA_COLS]
    val_time = x_val.index.to_numpy()

    train_set = sequential_window_dataset(
        x_train, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=N_SHIFT
    )
    val_set = sequential_window_dataset(
        x_val, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=N_SHIFT
    )

    """Create model"""
    model = build_bidirec_lstm_model(
        train_set, n_past=N_PAST, n_features=N_FEATURES, batch_size=BATCH_SIZE
    )

    """Train model"""
    early_stopping = keras.callbacks.EarlyStopping(patience=10)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    history = model.fit(
        train_set,
        epochs=EPOCHS,
        validation_data=val_set,
        callbacks=[early_stopping, tensorboard_callback],
    )
