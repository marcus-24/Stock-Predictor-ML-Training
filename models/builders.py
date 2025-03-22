from keras import (
    layers,
    models,
    optimizers,
    losses,
)  # TODO: figure out pylance errors
import tensorflow as tf

LR = 0.1


def scheduler(epoch: int) -> float:
    return LR * 10 ** (epoch / 10)


def build_model(
    data: tf.data.Dataset,
    n_features: int,
    n_labels: int,
    batch_size: int,
    win_size: int = 1,
) -> models.Sequential:
    """This function is used to build the forecasting model. Each iteration of creating a new
    model can be done in this function so that its tested in the unit tests before deployment
    """

    # TODO: Fine tune normalization layer. Performance is lower than scikit-learn standard scaler
    norm_layer = layers.Normalization()
    norm_layer.adapt(
        data.map(lambda x, _: x)
    )  # need to calculate the mean and variance for z-score (map used to extract only features and ignore labels)

    model = models.Sequential(
        [
            layers.InputLayer(shape=(win_size, n_features), batch_size=batch_size),
            norm_layer,
            layers.Bidirectional(layers.LSTM(5, return_sequences=True)),
            # layers.Bidirectional(layers.LSTM(5)),
            layers.Dense(n_labels),
        ]
    )

    optimizer = optimizers.SGD(learning_rate=LR, momentum=0.9)
    model.compile(loss=losses.Huber(), optimizer=optimizer, metrics=["mae"])
    model.summary()

    return model
