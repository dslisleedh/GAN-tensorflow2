import tensorflow as tf
import tensorflow_addons as tfa


'''
incomplete
'''


class Conv(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 activation='relu',
                 instance_norm=True
                 ):
        super(Conv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.instance_norm = instance_norm

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.filters,
                                   kernel_size=self.kernel_size,
                                   strides=self.strides,
                                   padding='same',
                                   activation='linear'
                                   )
        ])
        if self.instance_norm:
            self.forward.add(tfa.layers.InstanceNormalization())
        if self.activation == 'leakyrelu':
            self.forward.add(tf.keras.layers.LeakyReLU(.2))
        else:
            self.forward.add(tf.keras.layers.Activation(self.activation))

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(ResNetBlock, self).__init__()

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   activation='linear'
                                   ),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.Activation('relu')
        ])

    def call(self, inputs, **kwargs):
        return inputs + self.forward(inputs)


class TransConv(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1)
                 ):
        super(TransConv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=self.filters,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding='same',
                                            activation='linear'
                                            ),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.Activation('relu')
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class Generator(tf.keras.layers.Layer):
    def __init__(self):
        super(Generator, self).__init__()

        self.Downsampling = tf.keras.Sequential([
            Conv(64, (7, 7)),
            Conv(filters=128,
                 kernel_size=(4, 4),
                 strides=(2, 2)
                 ),
            Conv(filters=256,
                 kernel_size=(4, 4),
                 strides=(2, 2)
                 )
        ])
        self.Bottleneck = tf.keras.Sequential([
            ResNetBlock() for _ in range(6)
        ])
        self.Upsampling = tf.keras.Sequential([
            TransConv(filters=128,
                      kernel_size=(4, 4),
                      strides=(2, 2)
                      ),
            TransConv(filters=64,
                      kernel_size=(4, 4),
                      strides=(2, 2)
                      ),
            Conv(filters=3,
                 kernel_size=(7, 7),
                 strides=(1, 1),
                 activation='tanh',
                 instance_norm=False
                 )
        ])