import tensorflow as tf


class Deconv(tf.keras.layers.Layer):
    def __init__(self, filters, strides, bn=True, activation='lrelu'):
        super(Deconv, self).__init__()
        self.filters = filters
        self.strides = strides
        self.bn = bn
        self.activation = activation

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=self.filters,
                                            kernel_size=(3, 3),
                                            activation='linear',
                                            strides=self.strides,
                                            use_bias=False if self.bn else True,
                                            padding='same'
                                            )
        ])
        if self.bn:
            self.forward.add(tf.keras.layers.BatchNormalization())
        if self.activation == 'lrelu':
            self.forward.add(tf.keras.layers.LeakyReLU(.2))

    def call(self, X):
        y = self.forward(X)
        if self.activation == 'tanh':
            y = tf.nn.tanh(y)
        return y


class Generator(tf.keras.layers.Layer):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(7 * 7 * 256,
                                  use_bias=False,
                                  activation='linear'
                                  ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(.2)
        ])
        self.upsampling = tf.keras.Sequential([
            Deconv(filters=256,
                   strides=2
                   ),
            Deconv(filters=256,
                   strides=1
                   ),
            Deconv(filters=256,
                   strides=2
                   ),
            Deconv(filters=256,
                   strides=1
                   ),
            Deconv(filters=128,
                   strides=2
                   ),
            Deconv(filters=64,
                   strides=2
                   ),
            Deconv(filters=3,
                   strides=1,
                   bn=False,
                   activation='tanh'
                   )
        ])

    def call(self, latent):
        latent = self.fc(latent)
        return self.upsampling(tf.reshape(latent, (-1, 7, 7, 256)))


class Conv(tf.keras.layers.Layer):
    def __init__(self, filters, bn=True):
        super(Conv, self).__init__()
        self.filters = filters
        self.bn = bn

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.filters,
                                   kernel_size=(5, 5),
                                   strides=(2, 2),
                                   activation='linear',
                                   use_bias=False if self.bn else True,
                                   padding='same'
                                   )
        ])
        if self.bn:
            self.forward.add(tf.keras.layers.BatchNormalization())
        self.forward.add(tf.keras.layers.LeakyReLU(.2))

    def call(self, X):
        return self.forward(X)


class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.downsampling = tf.keras.Sequential([
            Conv(filters=64,
                 bn=False
                 ),
            Conv(filters=128),
            Conv(filters=256),
            Conv(filters=512)
        ])
        self.FC = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1,
                                  activation='linear'
                                  )
        ])

    def call(self, img):
        return tf.nn.tanh(self.FC(self.downsampling(img)))


class Lsgan(tf.keras.models.Model):
    def __init__(self, batch_size = 32):
        super(Lsgan, self).__init__()
        self.batch_size = batch_size

        self.Generator = Generator()
        self.Generator.build((None, 1024))
        self.Discriminator = Discriminator()
        self.Discriminator.build((None, 112, 112, 3))

    def compile(self, optimizer):
        super(Lsgan, self).compile()
        self.G_optimizer = optimizer
        self.D_optimizer = optimizer

    @tf.function
    def ls_disc_loss(self, true_disc, fake_disc):
        disc_loss = .5 * tf.reduce_mean(
            tf.losses.mean_squared_error(tf.ones_like(true_disc), true_disc)
        ) + .5 * tf.reduce_mean(
            tf.losses.mean_squared_error(tf.zeros_like(fake_disc), fake_disc)
        )
        return disc_loss

    @tf.functin
    def ls_gen_loss(self, fake_disc):
        gen_loss = .5 * tf.reduce_mean(
            tf.losses.mean_squared_error(tf.ones_like(fake_disc), fake_disc)
        )
        return gen_loss

    @tf.function
    def train_step(self, img):
        with tf.GradientTape(persistent=True) as tape:
            fake = self.Generator(tf.random.normal(shape=(self.batch_size, 1024)))
            disc_true = self.Discriminator(img)
            disc_fake = self.Discriminator(fake)
            disc_loss = self.ls_disc_loss(disc_true, disc_fake)
            gen_loss = self.ls_gen_loss(disc_fake)
        grads_d = tape.gradient(disc_loss, self.Discriminator.trainable_variables)
        self.D_optimizer.apply_gradients(
            zip(grads_d, self.Discriminator.trainable_variables)
        )
        grads_g = tape.gradient(gen_loss, self.Generator.trainable_variables)
        self.G_optimizer.apply_gradients(
            zip(grads_g, self.Generator.trainable_variables)
        )
        return {'discrimination_loss' : disc_loss, 'generation_loss' : gen_loss}

    @tf.function
    def call(self, latent):
        return self.Generator(latent)
