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

    def call(self, inputs, **kwargs):
        return self.forward(inputs) + inputs


class ReflectPadding2D(tf.keras.layers.Layer):
    def __init__(self, n_pad):
        super(ReflectPadding2D, self).__init__()
        self.n_pad = n_pad

    def call(self, inputs, **kwargs):
        return tf.pad(inputs,
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

    def call(self, inputs, **kwargs):
        y = self.downsampling(inputs)
        y = self.resblocks(y)
        return tf.nn.tanh(self.upsampling(y))


class DiscDownsamplingBlock(tf.keras.layers.Layer):
    def __init__(self, n_filters, strides=2):
        super(DiscDownsamplingBlock, self).__init__()
        self.n_filters = n_filters
        self.strides = strides

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.n_filters,
                                   kernel_size=(4, 4),
                                   strides=self.strides,
                                   padding='same',
                                   activation='linear'
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2)
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = 64,
                                   kernel_size = (4, 4),
                                   strides = (2, 2),
                                   padding = 'same',
                                   activation = tf.keras.layers.LeakyReLU(.2)
                                  ),
            DiscDownsamplingBlock(128),
            DiscDownsamplingBlock(256),
            DiscDownsamplingBlock(512, strides=1),
            tf.keras.layers.Conv2D(filters=1,
                                   kernel_size=(4, 4),
                                   activation='sigmoid',
                                   padding='same'
                                   )
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class history:
    def __init__(self, size=50):
        self.size = size
        self.hists = []

    def __call__(self, item):
        if len(self.hists) == 0:
            self.hists.append(item)
            return item
        else:
            self.hists.append(item)
            if len(self.hists) > self.size:
                self.hists = self.hists[-50:]
                return tf.concat(self.hists, axis=0)
            else:
                return tf.concat(self.hists, axis=0)


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
        self.Disc_x.build((None, 256, 256, 3))
        self.Disc_y.build((None, 256, 256, 3))
        self.G.build((None, 256, 256, 3))
        self.F.build((None, 256, 256, 3))
        self.hist_x = history()
        self.hist_y = history()

    def compile(self, optimizer):
        super(Cyclegan, self).compile()
        self.disc_optimizer = optimizer
        self.gen_optimizer = optimizer

    @tf.function
    def cycle_loss(self, real_image, recon_image):
        l1_loss = tf.reduce_mean(tf.losses.mean_absolute_error(real_image, recon_image))
        return self.lamb * l1_loss

    @tf.function
    def gen_loss(self, fake):
        gen_loss = .5 * tf.reduce_mean(tf.losses.mean_squared_error(fake, tf.ones_like(fake)))
        return gen_loss

    @tf.function
    def disc_loss(self, true, fake):
        disc_loss = .5 * (tf.reduce_mean(tf.losses.mean_squared_error(true, tf.ones_like(true))) +
                          tf.reduce_mean(tf.losses.mean_squared_error(fake, tf.zeros_like(fake)))
                          )
        return disc_loss

    @tf.function
    def update_discriminator(self, x, y, x_hat, y_hat):
        with tf.GradientTape(persistent=True) as tape:
            disc_y_true = self.Disc_y(y, training=True)
            disc_y_fake = self.Disc_y(y_hat, training=True)
            disc_x_true = self.Disc_x(x, training=True)
            disc_x_fake = self.Disc_x(x_hat, training=True)

            disc_y_loss = self.disc_loss(disc_y_true, disc_y_fake)
            disc_x_loss = self.disc_loss(disc_x_true, disc_x_fake)
            disc_loss = disc_y_loss + disc_x_loss
        grads = tape.gradient(disc_loss, self.Disc_y.trainable_variables + self.Disc_x.trainable_variables)
        self.disc_optimizer.apply_gradients(
            zip(grads, self.Disc_y.trainable_variables + self.Disc_x.trainable_variables)
        )
        return disc_x_loss, disc_y_loss

    @tf.function
    def update_generator(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self.G(x, training=True)
            x_hat = self.F(y, training=True)
            y_recon = self.G(x_hat, training=True)
            x_recon = self.F(y_hat, training=True)

            gen_g_loss = self.gen_loss(self.Disc_y(y_hat))
            gen_f_loss = self.gen_loss(self.Disc_x(x_hat))

            cycle_loss = self.cycle_loss(x, x_recon) + self.cycle_loss(y, y_recon)

            gen_loss = gen_g_loss + gen_f_loss + cycle_loss
            if self.identity_loss:
                identity_loss = .5 * (self.cycle_loss(y, self.G(y)) +
                                      self.cycle_loss(x, self.F(x))
                                      )
                gen_loss += identity_loss
        grads = tape.gradient(gen_loss, self.G.trainable_variables + self.F.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(grads, self.G.trainable_variables + self.F.trainable_variables)
        )
        return x_hat, y_hat, gen_g_loss, gen_f_loss, cycle_loss

    @tf.function
    def train_step(self, data):
        x, y = data

        x_hat, y_hat, gen_g_loss, gen_f_loss, cycle_loss = self.update_generator(x, y)
        x_hat = self.hist_x(x_hat)
        y_hat = self.hist_y(y_hat)
        disc_x_loss, disc_y_loss = self.update_discriminator(x, y, x_hat, y_hat)

        return {'disc_x_loss': disc_x_loss,
                'disc_y_loss': disc_y_loss,
                'g_loss': gen_g_loss,
                'f_loss': gen_f_loss,
                'cycle_loss' : cycle_loss
                }

    @tf.function
    def call(self, img):
        y_hat = self.G(img)
        X_hat = self.F(img)
        return y_hat, X_hat