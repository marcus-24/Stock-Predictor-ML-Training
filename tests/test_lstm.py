import pytest
import sys
import pandas as pd
import numpy as np
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off one DNN custom operations
from keras import models
import tensorflow as tf


from models.builders import build_model
from preprocessing.transformations import df_to_dataset

BATCH_SIZE = 32
N_PAST = 10
N_FUTURE = 10
N_FEATURES = 2
N_SHIFT = 1
N_LABELS = 1


@pytest.fixture
def train_set():
    x = np.arange(1000)
    y1 = np.sin(x)
    y2 = 2 * np.sin(0.5 * x)
    y3 = 2 * np.sin(0.5 * x + 0.5)
    df = pd.DataFrame(
        np.stack([y1, y2, y3], axis=1), columns=["feature 1", "feature 2", "label"]
    )
    return df_to_dataset(df, batch_size=BATCH_SIZE).repeat(2)


@pytest.fixture
def model(train_set):
    return build_model(
        train_set,
        n_features=N_FEATURES,
        batch_size=BATCH_SIZE,
        n_labels=N_LABELS,
    )


def test_lstm_convergence(model: models.Sequential, train_set: tf.data.Dataset):

    # Train the model
    history = model.fit(train_set, epochs=3, verbose=0)

    # Check if the loss decreased
    assert history.history["loss"][-1] < history.history["loss"][0]

    # Check if the accuracy increased
    assert history.history["mae"][-1] < history.history["mae"][0]


# def test_vanishing_gradient(model: models.Sequential, train_set: tf.data.Dataset):
#     pass


if __name__ == "__main__":
    pytest.main(sys.argv)
