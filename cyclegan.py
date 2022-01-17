import tensorflow as tf

class Resnetblock(tf.keras.layers.Layer):
    def __init__(self, n_filters=256):
        super(Resnetblock, self).__init__()
        self.n_filters = n_filters

        self.forward = tf.keras.Sequential([
            ReflectPadding2D(1),
            tf.keras.layers.Conv2D(filters=self.n_filters,
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   use_bias=False,
                                   activation='linear'
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            ReflectPadding2D(1),
            tf.keras.layers.Conv2D(filters=self.n_filters,
                                   kernel_size=(3, 3),
                                   padding='valid',
                                   use_bias=False,
                                   activation='linear'
                                   ),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, X):
        return self.forward(X) + X


class ReflectPadding2D(tf.keras.layers.Layer):
    def __init__(self, n_pad):
        super(ReflectPadding2D, self).__init__()
        self.n_pad = n_pad

    def call(self, X):
        return tf.pad(X,
                      paddings=[[0, 0],
                                [self.n_pad, self.n_pad],
                                [self.n_pad, self.n_pad],
                                [0, 0]
                                ],
                      mode='REFLECT'
                      )


class Generator(tf.keras.layers.Layer):
    def __init__(self, n_filters=64):
        super(Generator, self).__init__()
        self.n_filters = n_filters

        self.downsampling = tf.keras.Sequential([
            ReflectPadding2D(3),
            tf.keras.layers.Conv2D(filters=self.n_filters,
                                   kernel_size=(7, 7),
                                   activation='linear',
                                   use_bias=False,
                                   padding='valid'
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])
        for i in range(2):
            self.downsampling.add(tf.keras.layers.Conv2D(filters=self.n_filters * (2 ** (i + 1)),
                                                         kernel_size=(3, 3),
                                                         strides=(2, 2),
                                                         use_bias=False,
                                                         padding='same',
                                                         activation='linear'
                                                         )
                                  )
            self.downsampling.add(tf.keras.layers.BatchNormalization())
            self.downsampling.add(tf.keras.layers.ReLU())
        self.resblocks = tf.keras.Sequential(
            [Resnetblock() for _ in range(9)]
        )
        self.upsampling = tf.keras.Sequential()
        for i in range(2):
            self.upsampling.add(tf.keras.layers.Conv2DTranspose(filters=self.n_filters * (2 ** (1 - i)),
                                                                kernel_size=(3, 3),
                                                                strides=(2, 2),
                                                                padding='same',
                                                                use_bias='False'
                                                                )
                                )
            self.upsampling.add(tf.keras.layers.BatchNormalization())
            self.upsampling.add(tf.keras.layers.ReLU())
        self.upsampling.add(ReflectPadding2D(3))
        self.upsampling.add(tf.keras.layers.Conv2D(filters=3,
                                                   kernel_size=(7, 7),
                                                   activation='linear',
                                                   padding='valid'
                                                   )
                            )

    def call(self, X):
        y = self.downsampling(X)
        y = self.resblocks(y)
        return tf.nn.tanh(self.upsampling(y))


class DiscDownsamplingBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, strides=2):
        super(DiscDownsamplingBlock, self).__init__()
        self.n_filters = n_filters
        self.strides = strides

        self.forward = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(filters=self.n_filters,
                                   kernel_size=(4, 4),
                                   strides=self.strides,
                                   padding='valid',
                                   activation='linear'
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2)
        ])

    def call(self, X):
        return self.forward(X)


class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.forward = tf.keras.Sequential([
            DiscDownsamplingBlock(64),
            DiscDownsamplingBlock(128),
            DiscDownsamplingBlock(256),
            DiscDownsamplingBlock(512, strides=1),
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(filters=1,
                                   kernel_size=(4, 4),
                                   activation='sigmoid',
                                   padding='valid'
                                   )
        ])

    def call(self, X):
        return self.forward(X)


class Cyclegan(tf.keras.models.Model):
    '''
    Must set batch_size as 1
    '''
    def __init__(self, lamb=10., identity_loss=False):
        super(Cyclegan, self).__init__()
        self.lamb = lamb
        self.identity_loss = identity_loss

        self.G = Generator()
        self.F = Generator()
        self.Disc_x = Discriminator()
        self.Disc_y = Discriminator()

    def compile(self, lr):
        super(Cyclegan, self).compile()
        self.disc_x_optimizer = tf.keras.optimizers.Adam(lr, beta_1=.5)
        self.disc_y_optimizer = tf.keras.optimizers.Adam(lr, beta_1=.5)
        self.g_optimizer = tf.keras.optimizers.Adam(lr, beta_1=.5)
        self.f_optimizer = tf.keras.optimizers.Adam(lr, beta_1=.5)

    @tf.function
    def train_step(self, data):
        X, y = data

        with tf.GradientTape(persistent=True) as tape:
            y_hat = self.G(X)
            X_recon = self.F(y_hat)
            X_hat = self.F(y)
            y_recon = self.G(X_hat)
            disc_y_true = self.Disc_y(y)
            disc_y_fake = self.Disc_y(y_hat)
            disc_x_true = self.Disc_x(X)
            disc_x_fake = self.Disc_x(X_hat)

            disc_y_loss = .5 * (tf.reduce_mean(
                tf.square(disc_y_true - tf.ones_like(disc_y_true))
            ) + tf.reduce_mean(
                tf.square(disc_y_fake)
            ))
            disc_x_loss = .5 * (tf.reduce_mean(
                tf.square(disc_x_true - tf.ones_like(disc_x_true))
            ) + tf.reduce_mean(
                tf.square(disc_x_fake)
            ))
            gen_g_loss = tf.reduce_mean(
                tf.square(disc_y_fake - tf.ones_like(disc_y_fake))
            ) + self.lamb * tf.reduce_mean(
                tf.abs(X - X_recon)
            )
            gen_f_loss = tf.reduce_mean(
                tf.square(disc_x_fake - tf.ones_like(disc_x_fake))
            ) + self.lamb * tf.reduce_mean(
                tf.abs(y - y_recon)
            )
            if self.identity_loss:
                gen_g_loss += self.lamb * .5 * tf.reduce_mean(
                    tf.abs(y, self.G(y))
                )
                gen_f_loss += self.lamb * .5 * tf.reduce_mean(
                    tf.abs(X, self.F(X))
                )

        grads = tape.gradient(disc_y_loss, self.Disc_y.trainable_variables)
        self.disc_y_optimizer.apply_gradients(
            zip(grads, self.Disc_y.trainable_variables)
        )
        grads = tape.gradient(disc_x_loss, self.Disc_x.trainable_variables)
        self.disc_x_optimizer.apply_gradients(
            zip(grads, self.Disc_x.trainable_variables)
        )
        grads = tape.gradient(gen_g_loss, self.G.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(grads, self.G.trainable_variables)
        )
        grads = tape.gradient(gen_f_loss, self.F.trainable_variables)
        self.f_optimizer.apply_gradients(
            zip(grads, self.F.trainable_variables)
        )

        return {'Disc_X_loss': disc_x_loss,
                'Disc_y_loss': disc_y_loss,
                'G_loss': gen_g_loss,
                'F_loss': gen_f_loss
                }

    @tf.function
    def call(self, X):
        y_hat = self.G(X)
        X_hat = self.F(X)
        _ = self.Disc_y(X)
        _ = self.Disc_x(X)
        return y_hat, X_hat