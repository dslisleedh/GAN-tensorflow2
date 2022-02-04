import tensorflow as tf


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, n_channels, mode=None, bn: bool = False, identity=True):
        super(ResnetBlock, self).__init__()
        self.n_channels = n_channels
        self.mode = mode
        self.bn = bn
        self.identity = identity

        if mode == 'up':
            self.forward = tf.keras.Sequential([
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(filters=self.n_channels,
                                       kernel_size=(3, 3),
                                       padding='same',
                                       activation='linear',
                                       kernel_initializer=tf.keras.initializers.random_normal(stddev=.02),
                                       use_bias=False
                                       ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.UpSampling2D(size=(2, 2),
                                             interpolation='nearest'
                                             ),
                tf.keras.layers.Conv2D(filters=self.n_channels,
                                       kernel_size=(3, 3),
                                       padding='same',
                                       activation='linear',
                                       kernel_initializer=tf.keras.initializers.random_normal(stddev=.02),
                                       use_bias=False
                                       )
            ])
            self.skip = tf.keras.layers.UpSampling2D(size=(2, 2),
                                                     interpolation='nearest'
                                                     )
        else:
            self.forward = tf.keras.Sequential()
            for _ in range(2):
                self.forward.add(tf.keras.layers.ReLU())
                self.forward.add(tf.keras.layers.Conv2D(filters=self.n_channels,
                                                        kernel_size=(3, 3),
                                                        padding='same',
                                                        activation='linear',
                                                        kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                                        )
                                 )
            if self.mode == 'down':
                self.forward.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                                  strides=(2, 2),
                                                                  padding='valid'
                                                                  )
                                 )
            if self.mode == 'down':
                self.skip = tf.keras.Sequential([
                    tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                     strides=(2, 2),
                                                     padding='valid'
                                                     )
                ])
                if not self.identity:
                    self.skip.add(tf.keras.layers.Conv2D(filters= self.n_channels,
                                                         kernel_size=(1, 1),
                                                         padding='valid',
                                                         activation='linear',
                                                         kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02),
                                                         use_bias=False
                                                         )
                                  )
            else:
                self.skip = tf.keras.layers.Layer()

    def call(self, inputs, **kwargs):
        return self.forward(inputs) + self.skip(inputs)


class Generator(tf.keras.layers.Layer):
    def __init__(self, z_dims):
        super(Generator, self).__init__()
        self.z_dims = z_dims

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * 4 * self.z_dims,
                                  kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                  ),
            tf.keras.layers.Reshape((4, 4, self.z_dims)),
            # 4
            ResnetBlock(self.z_dims,
                        mode='up',
                        bn=True
                        ),
            # 8
            ResnetBlock(self.z_dims,
                        mode='up',
                        bn=True
                        ),
            # 16
            ResnetBlock(self.z_dims,
                        mode='up',
                        bn=True
                        ),
            # 32
            tf.keras.layers.Conv2D(3,
                                   kernel_size=(3, 3),
                                   padding='same',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=.02),
                                   activation='tanh'
                                   )
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class Critic(tf.keras.layers.Layer):
    def __init__(self, z_dims):
        super(Critic, self).__init__()
        self.z_dims = z_dims

        self.forward = tf.keras.Sequential([
            # 32
            ResnetBlock(self.z_dims,
                        mode='down',
                        identity=False
                        ),
            # 16
            ResnetBlock(self.z_dims,
                        mode='down'
                        ),
            # 8
            ResnetBlock(self.z_dims),
            # 8
            ResnetBlock(self.z_dims),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D(),
            # 1
            tf.keras.layers.Dense(1,
                                  kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                  )
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class WganGp(tf.keras.models.Model):
    '''
    Wasserstein GAN - Gradient Penalty
    '''

    def __init__(self,
                 lamb=10.,
                 n_critic=5,
                 alpha=2 * 1e-4,
                 beta_1=0.,
                 beta_2=.9,
                 dim_latent=128,
                 batch_size=64
                 ):
        super(WganGp, self).__init__()
        self.lamb = lamb
        self.n_critic = n_critic
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.dim_latent = dim_latent
        self.batch_size = batch_size

        self.Generator = Generator(self.dim_latent)
        self.Generator.build((None, self.dim_latent))
        self.Critic = Critic(self.dim_latent)
        self.Critic.build((None, 32, 32, 3))
        self.compile()
        self.hist = []

    def compile(self):
        super(WganGp, self).compile()
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha,
                                                    beta_1=self.beta_1,
                                                    beta_2=self.beta_2,
                                                    decay=self.alpha/100000
                                                    )
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha,
                                                    beta_1=self.beta_1,
                                                    beta_2=self.beta_2,
                                                    decay=self.alpha/100000
                                                    )

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
        epsilon = tf.random.uniform(minval=0.,
                                    maxval=1.,
                                    shape=(self.batch_size, 1, 1, 1)
                                    )
        x_hat = epsilon * x + (1 - epsilon) * x_tilde
        avg_logit = self.Critic(x_hat, training=True)
        grads = tf.keras.backend.gradients(avg_logit, [x_hat])[0]
        l2norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean(tf.square(l2norm - 1.0))
        return gp

    @tf.function
    def train_step(self, data):
        data = tf.split(data,
                        num_or_size_splits=self.n_critic
                        )
        # 1. Update Critic for self.n_critic times
        mean_criticism_loss = 0.
        for i in range(self.n_critic):
            x = data[i]
            with tf.GradientTape() as tape:
                x_tilde = self.Generator(tf.random.normal((self.batch_size, self.dim_latent)))
                gp = self.compute_gp(x, x_tilde)
                loss = self.compute_critic_loss(self.Critic(x, training=True),
                                                self.Critic(x_tilde, training=True)
                                                )
                loss += gp * self.lamb
            grads = tape.gradient(loss, self.Critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(grads, self.Critic.trainable_variables)
            )
            mean_criticism_loss += loss
        mean_criticism_loss /= self.n_critic

        # 2. Update Generator
        with tf.GradientTape() as tape:
            x_tilde = self.Generator(tf.random.normal((self.batch_size, self.dim_latent)),
                                     training=True
                                     )
            loss = self.compute_gen_loss(self.Critic(x_tilde))
        grads = tape.gradient(loss, self.Generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(grads, self.Generator.trainable_variables)
        )

        return {'mean_criticism_loss': mean_criticism_loss, 'generation_loss': loss}

    @tf.function
    def call(self, inputs):
        a = self.Critic(inputs)
        return self.Generator(tf.random.normal((1, self.dim_latent)))