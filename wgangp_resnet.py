import tensorflow as tf


class Downsampling2D(tf.keras.layers.Layer):
    def __init__(self):
        super(Downsampling2D, self).__init__()

        self.forward = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                        strides=(2, 2),
                                                        padding='valid'
                                                        )

    def call(self, inputs, **kwargs):
        return self.forward(inputs)

class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, n_channels, mode=None, bn:bool=False):
        super(ResnetBlock, self).__init__()
        self.n_channels = n_channels
        self.mode = mode
        self.bn = bn

        self.forward = tf.keras.Sequential()
        for _ in range(2):
            if self.bn:
                self.forward.add(tf.keras.layers.BatchNormalization())
            self.forward.add(tf.keras.layers.ReLU())
            self.forward.add(tf.keras.layers.Conv2D(filters=self.n_channels,
                                                    kernel_size=(3, 3),
                                                    padding='same',
                                                    kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                                    )
                             )
        if self.mode == 'down':
            self.scaling = Downsampling2D()
        elif self.mode == 'up':
            self.scaling = tf.keras.layers.UpSampling2D(size=(2, 2),
                                                        interpolation='nearest'
                                                        )
        else:
            self.scaling = tf.keras.layers.Layer()

    def call(self, inputs, **kwargs):
        return self.scaling(self.forward(inputs) + inputs)