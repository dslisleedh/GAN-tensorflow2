import tensorflow as tf


class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, c):
        self.c = c

    def call(self, weights):
        return tf.keras.backend.clip(weights, -self.clip_value, self.clip_value)


class Wgan(tf.keras.models.Model):
    def __init__(self):
        super(Wgan, self).__init__()
        self.alpha = .00005
        self.c = 0.01
        self.Const = ClipConstraint(self.c)
        self.n_critic = 5

        self.Generator = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * 4 * 1024, activation='linear',
                                  kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                                  kernel_constraint=self.Const
                                  ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Reshape((4, 4, 1024)),
            tf.keras.layers.Conv2DTranspose(filters=512,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            padding='same',
                                            activation='linear',
                                            kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                                            kernel_constraint=self.Const
                                            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=256,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            padding='same',
                                            activation='linear',
                                            kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                                            kernel_constraint=self.Const,
                                            use_bias=False
                                            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=128,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            padding='same',
                                            activation='linear',
                                            kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                                            kernel_constraint=self.Const
                                            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(filters=1,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            padding='same',
                                            activation='tanh',
                                            kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                                            kernel_constraint=self.Const
                                            )
        ])
        self.Generator.build((None, 100))
        self.Critic = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(5, 5),
                                   padding='same',
                                   strides=(2, 2),
                                   activation='linear',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                                   kernel_constraint=self.Const
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(5, 5),
                                   padding='same',
                                   strides=(2, 2),
                                   activation='linear',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                                   kernel_constraint=self.Const
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=(5, 5),
                                   padding='same',
                                   strides=(2, 2),
                                   activation='linear',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                                   kernel_constraint=self.Const
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(filters=512,
                                   kernel_size=(5, 5),
                                   padding='same',
                                   strides=(2, 2),
                                   activation='linear',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                                   kernel_constraint=self.Const
                                   ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(filters=1,
                                   kernel_size=(4, 4),
                                   activation='sigmoid',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
                                   kernel_constraint=self.Const,
                                   padding='valid'
                                   ),
            tf.keras.layers.Flatten()
        ])
        self.Critic.build((None, 64, 64, 1))
        self.compile()
        self.Resize = tf.keras.layers.experimental.preprocessing.Resizing(height=64, width=64)

    def compile(self):
        super(Wgan, self).compile()
        self.g_optimizer = tf.keras.optimizers.RMSprop(self.alpha)
        self.c_optimizer = tf.keras.optimizers.RMSprop(self.alpha)

    @tf.function
    def compute_disc_loss(self, true, fake):
        loss = tf.reduce_mean(
            true
        ) - tf.reduce_mean(
            fake
        )
        return loss

    @tf.function
    def compute_gen_loss(self, fake):
        loss = tf.reduce_mean(fake)
        return loss

    @tf.function
    def train_step(self, img):
        true_label = -tf.ones((tf.shape(img)[0], 1))
        fake_label = tf.ones((tf.shape(img)[0], 1))
        img = self.Resize(img)

        disc_mean_loss = 0
        for _ in range(self.n_critic):
            fake_img = self.Generator(tf.random.normal((tf.shape(img)[0], 100)), training=False)
            with tf.GradientTape() as tape:
                true_logit = self.Critic(img, training=True)
                fake_logit = self.Critic(fake_img, training=True)
                loss = self.compute_disc_loss(true_logit, fake_logit)
            grads = tape.gradient(loss, self.Critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(grads, self.Critic.trainable_variables)
            )
            disc_mean_loss += loss
        disc_mean_loss /= self.n_critic

        with tf.GradientTape() as tape:
            fake_img = self.Generator(tf.random.normal((tf.shape(img)[0], 100)), training=True)
            fake_logit = self.Critic(fake_img, training=False)
            gen_loss = self.compute_gen_loss(fake_logit)
        grads = tape.gradient(gen_loss, self.Generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(grads, self.Generator.trainable_variables)
        )

        return {'mean_discrimination_loss': disc_mean_loss, 'generation_loss': gen_loss}

