import tensorflow as tf


class TransConv(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(TransConv, self).__init__()
        self.filters = filters

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters = self.filters,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            use_bias=False,
                                            padding='same',
                                            kernel_initializer=tf.keras.initializers.random_normal(stddev=.02),
                                            activation='linear'
                                            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class Generator(tf.keras.layers.Layer):
    def __init__(self):
        super(Generator, self).__init__()

        self.upsampling = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * 4 * 1024,
                                  activation='linear',
                                  kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                  ),
            tf.keras.layers.Reshape((4, 4, 1024)),
            TransConv(512),
            TransConv(256),
            TransConv(128),
            tf.keras.layers.Conv2DTranspose(3,
                                            kernel_size=(5, 5),
                                            activation='linear',
                                            kernel_initializer=tf.keras.initializers.random_normal(stddev=.02),
                                            padding='same',
                                            strides=(2, 2)
                                            )
        ])

    def call(self, inputs, **kwargs):
        return tf.nn.tanh(self.upsampling(inputs))


class Conv(tf.keras.layers.Layer):
    def __init__(self, filters, bn=True):
        super(Conv, self).__init__()
        self.filters = filters
        self.bn = bn

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.filters,
                                   kernel_size=(4, 4),
                                   activation='linear',
                                   padding='same',
                                   strides=(2,2),
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=.02),
                                   use_bias=False if self.bn else True
                                   )
            ])
        if self.bn:
            self.forward.add(tf.keras.layers.BatchNormalization())
        self.forward.add(tf.keras.layers.LeakyReLU(.2))

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class Discriminator(tf.keras.layers.Layer):
    '''
    There's no specific description of discriminator in original paper.
    So I made discriminator based on pytorch official tutorial.
    '''
    def __init__(self):
        super(Discriminator, self).__init__()

        self.forward = tf.keras.Sequential([
            Conv(filters=64,
                 bn=False
                 ),
            Conv(128),
            Conv(256),
            Conv(512),
            tf.keras.layers.Conv2D(filters=1,
                                   kernel_size=(4,4),
                                   padding='valid',
                                   activation='sigmoid',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                   ),
            tf.keras.layers.Flatten()
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class Dcgan(tf.keras.models.Model):
    def __init__(self):
        super(Dcgan, self).__init__()

        self.Generator = Generator()
        self.Discriminator = Discriminator()

    def compile(self, optimizer):
        super(Dcgan, self).compile()
        self.D_optimizer = optimizer
        self.G_optimizer = optimizer

    @tf.function
    def compute_disc_loss(self, true, fake):
        disc_loss = tf.reduce_mean(
            tf.losses.binary_crossentropy(tf.ones_like(true), true)
        ) + tf.reduce_mean(
            tf.losses.binary_crossentropy(tf.zeros_like(fake), fake)
        )
        return disc_loss

    @tf.function
    def compute_gen_loss(self, fake):
        gen_loss = tf.reduce_mean(
            tf.losses.binary_crossentropy(tf.ones_like(fake), fake)
        )
        return gen_loss

    @tf.function
    def train_step(self, img):
        fake = self.Generator(img)

        # 1. Update discriminator
        with tf.GradientTape() as tape:
            disc_true = self.Discriminator(img)
            disc_fake = self.Discriminator(fake)
            disc_loss = self.compute_disc_loss(disc_true, disc_fake)
        grads = tape.gradient(disc_loss, self.Discriminator.trainable_variables)
        self.D_optimizer.apply_gradients(
            zip(grads, self.Discriminator.trainable_variables)
        )

        # 2. Update generator
        with tf.GradientTape() as tape:
            gen_loss = self.compute_gen_loss(self.Discriminator(img))
        grads = tape.gradient(gen_loss, self.Generator.trainable_variables)
        self.G_optimizer.apply_gradients(
            zip(grads, self.Generator.trainable_variables)
        )

        return {'discrimination_loss' : disc_loss, 'generation_loss' : gen_loss}

    def call(self, inputs, training=None, mask=None):
        return self.Generator(inputs)
