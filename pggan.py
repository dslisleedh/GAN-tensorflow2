import tensorflow as tf
import numpy as np


'''
incomplete
'''


class PixelNormalization(tf.keras.layers.Layer):
    def __init__(self, eps=1e-8):
        super(PixelNormalization, self).__init__()
        self.eps = eps

    def call(self, inputs, **kwargs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.eps)


class UpscalingConv2D(tf.keras.layers.Layer):
    def __init__(self, n_filters, initial=False):
        super(UpscalingConv2D, self).__init__()
        self.n_filters = tf.minimum(n_filters, 512)
        self.initial = initial

        if self.initial:
            self.forward = tf.keras.Sequential([
                tf.keras.layers.Dense(units=self.n_filters * 4 * 4,
                                      kernel_initializer='he_normal'
                                      ),
                tf.keras.layers.Reshape((4, 4, self.n_filters)),
                PixelNormalization(),
                tf.keras.layers.Conv2D(filters=self.n_filters,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same',
                                       kernel_initializer='he_normal',
                                       activation=tf.keras.layers.LeakyReLU(.2)
                                       ),
                PixelNormalization()
            ])
        else:
            self.forward = tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(filters=self.n_filters,
                                                kernel_size=(3, 3),
                                                strides=(2, 2),
                                                padding='same',
                                                kernel_initializer='he_normal',
                                                activation=tf.keras.layers.LeakyReLU(.2)
                                                ),
                PixelNormalization(),
                tf.keras.layers.Conv2D(filters=self.n_filters,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same',
                                       kernel_initializer='he_normal',
                                       activation=tf.keras.layers.LeakyReLU(.2)
                                       ),
                PixelNormalization()
            ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class DownsamplingConv2D(tf.keras.layers.Layer):
    def __init__(self, filters_in, last=False):
        super(DownsamplingConv2D, self).__init__()
        self.filters_in = tf.minimum(512, filters_in)
        self.filters_out = tf.minimum(512, self.filters_in * 2)
        self.last = last

        if self.last:
            self.forward = tf.keras.Sequential([
                MinibatchStddev(),
                tf.keras.layers.Conv2D(filters=self.filters_in,
                                       kernel_size=(3, 3),
                                       strides=(1,1),
                                       padding='same',
                                       kernel_initializer='he_normal',
                                       activation=tf.keras.layers.LeakyReLU(.2)
                                       ),
                tf.keras.layers.Conv2D(filters=self.filters_out,
                                       kernel_size=(3, 3),
                                       strides=(2, 2),
                                       padding='same',
                                       kernel_initializer='he_normal',
                                       activation=tf.keras.layers.LeakyReLU(.2)
                                       ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1)
            ])
        else:
            self.forward = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters=self.filters_in,
                                       kernel_size=(3, 3),
                                       strides=(1, 1),
                                       padding='same',
                                       kernel_initializer='he_normal',
                                       activation=tf.keras.layers.LeakyReLU(.2)
                                       ),
                tf.keras.layers.Conv2D(filters=self.filters_out,
                                       kernel_size=(3, 3),
                                       stridies=(2, 2),
                                       padding='same',
                                       kernel_initializer='he_normal',
                                       activation=tf.keras.layers.LeakyReLU(.2)
                                       )
            ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class MinibatchStddev(tf.keras.layers.Layer):
    def __init__(self, group_size=4, epsilon=1e-8):
        super(MinibatchStddev, self).__init__()
        self.group_size = group_size
        self.epslion = epsilon

    def call(self, inputs, **kwargs):
        shape = tf.shape(inputs)
        g = tf.minimum(self.group_size, shape[0])
        y = tf.reshape(inputs, (g, -1,  shape[1], shape[2], shape[3]))
        y -= tf.reduce_mean(y,
                            axis=0,
                            keepdims=True
                            )
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + self.epslion)
        y = tf.reduce_mean(y,
                           axis=[1,2,3],
                           keepdims=True
                           )
        y = tf.tile(y, (g, shape[1], shape[2], 1))
        return tf.concat([inputs, y], axis=-1)


class Downsampling2D(tf.keras.layers.Layer):
    def __init__(self, size):
        super(Downsampling2D, self).__init__()
        if len(size) == 1:
            self.size = (size, size)
        elif len(size) > 2:
            raise ValueError('Downsampling2D only supports 2-dimensional scaling')
        else:
            self.size = size

        self.forward = tf.keras.layers.AveragePooling2D(pool_size=self.size,
                                                        strides=self.size,
                                                        padding='valid'
                                                        )

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class ToRGB(tf.keras.layers.Layer):
    def __init__(self):
        super(ToRGB, self).__init__()

        self.forward = tf.keras.layers.Conv2D(filters = 3,
                                              kernel_size=(1, 1),
                                              strides=(1, 1),
                                              padding='valid',
                                              )

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class FromRGB(tf.keras.layers.Layer):
    def __init__(self, n_channels):
        super(FromRGB, self).__init__()
        self.n_channels = n_channels
        self.forward = tf.keras.layers.Conv2D(filters=self.n_channels,
                                              kernel_size=(1, 1),
                                              strides=(1, 1),
                                              padding='valid'
                                              )

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class Pggan(tf.keras.models.Model):
    def __init__(self,
                 build_datasets,
                 output_channel=3,
                 output_size=512,
                 dim_latent=256
                 ):
        super(Pggan, self).__init__()
        self.output_channel = output_channel
        n_upsampling = int((np.log(output_size) / np.log(2)) - 1)
        if (n_upsampling == 0) | (n_upsampling % 1 != 0):
            assert ValueError('size must power of 2')
        else:
            self.output_size = output_size
            self.n_layers = n_upsampling
        self.dim_latent = dim_latent

        self.Generator, self.Critic, g_optimizer, c_optimizer = self.progressive_build(build_datasets)

    def compile(self, **kwargs):
        super(Pggan, self).compile(**kwargs)

    ####### 여기수정
    @tf.function
    def progressive_build(self, build_dataset):
        # Build Generator
        print('Growing models')
        print('------------------------------------------------------------------------------')
        generator = [UpscalingConv2D(int(np.power(2., self.n_layers-1)), initial=True)]
        for i in range(self.n_layers-1):
            layer = [UpscalingConv2D(16 * (2**((self.n_layers-1) - (i + 1))))]
            if i == self.n_layers-2:
                layer += tf.keras.layers.Conv2D(filters=3,
                                                kernel_size=(1, 1),
                                                strides=(1, 1),
                                                padding='valid',
                                                kernel_initializer='he_normal'
                                                )
                layer = tf.keras.Sequential(layer)
            generator.append(layer)
        # Build Critic
        critic = [tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=16,
                                   kernel_size=(1,1),
                                   strides=(1, 1),
                                   padding='valid',
                                   kernel_initializer='he_normal',
                                   activation=tf.keras.layers.LeakyReLU(.2)
                                   ),
            DownsamplingConv2D(16)
        ])]
        for i in range(self.n_layers-1):
            if i == self.n_layers-2:
                critic.append(DownsamplingConv2D(512, last=True))
            else:
                critic.append(DownsamplingConv2D(32 * (2**i)))

        ### progressive growing build 학습과정 작성하면됨.

        for i in range(self.n_layers):
            print(f'Building resolution {2**(i + 2)}x{2**(i + 2)}')

        g_optimizer = []
        c_optimizer = []
        return generator, critic, g_optimizer, c_optimizer

    @tf.function
    def compute_critic_loss(self, true_logit, fake_logit):
        loss = tf.reduce_mean(
            fake_logit
        ) - tf.reduce_mean(
            true_logit
        )
        return loss

    @tf.function
    def compute_gen_loss(self, fake_logit):
        loss = -tf.reduce_mean(
            fake_logit
        )
        return loss

    @tf.function
    def compute_gp(self, x, x_tilde):
        epsilon = tf.random.uniform(shape=(tf.shape(x)[0], 1, 1, 1),
                                    minval=0.,
                                    maxval=1.
                                    )
        x_hat = epsilon * x + (1 - epsilon) * x_tilde
        avg_logit = self.Critic(x_hat, training=True)
        grads = tf.keras.backend.gradients(avg_logit, [x_hat])[0]
        l2norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean(tf.square(l2norm - 1.0))
        return gp