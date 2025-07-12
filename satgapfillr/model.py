import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate

class UNetGapFiller:
    def __init__(self, input_shape):
        self.model = self._build(input_shape)

    def _build(self, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Encoder
        conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D()(conv1)

        conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D()(conv2)

        # Bottleneck
        conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)

        # Decoder
        up1 = UpSampling2D()(conv3)
        concat1 = concatenate([up1, conv2])
        conv4 = Conv2D(64, 3, activation='relu', padding='same')(concat1)
        conv4 = Conv2D(64, 3, activation='relu', padding='same')(conv4)

        up2 = UpSampling2D()(conv4)
        concat2 = concatenate([up2, conv1])
        conv5 = Conv2D(32, 3, activation='relu', padding='same')(concat2)
        conv5 = Conv2D(32, 3, activation='relu', padding='same')(conv5)

        outputs = Conv2D(1, 1, activation='linear')(conv5)

        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')  # swap with custom loss later if you want
        return model

    def fit(self, dataset, **kwargs):
        self.model.fit(dataset, **kwargs)

    def predict(self, X):
        return self.model.predict(X)
