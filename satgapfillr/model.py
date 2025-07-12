import tensorflow as tf

class UNetGapFiller:
    def __init__(self, input_shape):
        self.model = self._build(input_shape)

    def _build(self, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)
        outputs = inputs  # TODO: replace with UNet blocks
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, dataset, **kwargs):
        self.model.fit(dataset, **kwargs)

    def predict(self, X):
        return self.model.predict(X)
