from keras import layers, models, optimizers, losses  # TODO: figure out pylance errors
import tensorflow as tf


def build_model(
    data: tf.data.Dataset, n_past: int, n_features: int, batch_size: int
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
