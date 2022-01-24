import tensorflow as tf


class Generator(tf.keras.layers.Layer):
    def __init__(self):
        super(Generator, self).__init__()
        self.ki = tf.keras.initializers.random_normal(stddev=.02)

        self.upsampling = tf.keras.Sequential([
            tf.keras.layers.Dense(7 * 7 * 256,
                                  activation='linear',
                                  kernel_initializer=self.ki,
                                  use_bias=False
                                  ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Reshape((7, 7, 256)),
            tf.keras.layers.Conv2DTranspose(filters=256,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            activation='linear',
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=self.ki
                                            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=128,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            activation='linear',
                                            padding='same',
                                            use_bias=False,
                                            kernel_initializer=self.ki
                                            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=1,
                                            kernel_size=(5, 5),
                                            activation='tanh',
                                            padding='same',
                                            kernel_initializer=self.ki
                                            )
        ])

    def call(self, inputs, **kwargs):
        return self.upsampling(inputs)


class Critic(tf.keras.layers.Layer):
    def __init__(self):
        super(Critic, self).__init__()
        self.ki = tf.keras.initializers.random_normal(stddev=.02)

        self.downsampling = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(5, 5),
                                   padding='same',
                                   strides=(2, 2),
                                   activation='linear',
                                   kernel_initializer=self.ki
                                   ),
            tf.keras.layers.LeakyReLU(.2),
            tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(5, 5),
                                   padding='same',
                                   strides=(2, 2),
                                   activation='linear',
                                   kernel_initializer=self.ki,
                                   use_bias=False
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(.2),
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(5, 5),
                                   padding='same',
                                   strides=(2, 2),
                                   activation='linear',
                                   kernel_initializer=self.ki,
                                   use_bias=False
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(.2),
            tf.keras.layers.Conv2D(filters=1,
                                   kernel_size=(4, 4),
                                   activation='linear',
                                   kernel_initializer=self.ki,
                                   padding='valid'
                                   ),
            tf.keras.layers.Flatten()
        ])

    def call(self, inputs, **kwargs):
        return self.downsampling(inputs)


class Wgan(tf.keras.models.Model):
    '''
    set train batch_size as 64 * n_critic
    '''
    def __init__(self):
        super(Wgan, self).__init__()
        self.alpha = .00005
        self.c = 0.01
        self.n_critic = 5

        self.Generator = Generator()
        self.Generator.build((None, 100))
        self.Critic = Critic()
        self.Critic.build((None, 28, 28, 1))
        self.compile()
        self.hist = []

    def compile(self):
        super(Wgan, self).compile()
        self.g_optimizer = tf.keras.optimizers.RMSprop(-self.alpha)
        self.c_optimizer = tf.keras.optimizers.RMSprop(self.alpha)

    @tf.function
    def train_step(self, img):
        img = tf.split(img,
                       axis=0,
                       num_or_size_splits=self.n_critic
                       )

        # 1. update discriminator
        mean_criticism_loss = 0
        for i in range(self.n_critic):
            with tf.GradientTape() as tape:
                # Sample x from real data
                x = img[i]
                # Sample z from prior
                z = tf.random.normal(shape=(tf.shape(x)[0], 100))
                # Compute grads_w : [ 1/m * sigma^m_i=1 f_w(x^i) - 1/m * sigma^m_i=1 f_w(g_theta(z^i)) ]
                loss = tf.reduce_mean(
                    self.Critic(x, training=True)
                ) - tf.reduce_mean(
                    self.Critic(self.Generator(z, training=False), training=True)
                )
            grads_w = tape.gradient(loss, self.Critic.trainable_variables)
            # Gradient ascent(-alpha)
            self.c_optimizer.apply_gradients(
                zip(grads_w, self.Critic.trainable_variables)
            )
            W = self.Critic.trainable_variables
            [tf.compat.v1.assign(w, tf.clip_by_value(w, -self.c, self.c)) for w in W]
            mean_criticism_loss += loss

        # 2. update generator
        with tf.GradientTape() as tape:
            # Sample z from prior
            z = tf.random.normal(shape=(tf.shape(x)[0], 100))
            # Compute grads_theta : - 1/m * sigma^m_i=1 f_w(g_theta(z^i))
            loss = -tf.reduce_mean(self.Critic(self.Generator(z, training=True), training=False))
        grads_theta = tape.gradient(loss, self.Generator.trainable_variables)
        # Gradient descent
        self.g_optimizer.apply_gradients(
            zip(grads_theta, self.Generator.trainable_variables)
        )

        return {'mean_criticism_loss': mean_criticism_loss / self.n_critic, 'generation_loss': loss}