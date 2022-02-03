import tensorflow as tf
import tensorflow_addons as tfa


class UpsamplingConv2D(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super(UpsamplingConv2D, self).__init__()
        self.n_filters = n_filters

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters=self.n_filters,
                                            kernel_size=(5, 5),
                                            strides=(2, 2),
                                            padding='same',
                                            activation='linear',
                                            kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02),
                                            use_bias=False
                                            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class Generator(tf.keras.layers.Layer):
    def __init__(self, dims=64):
        super(Generator, self).__init__()
        self.dims = dims

        self.latent_projection = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dims * 8 * 4 * 4,
                                  use_bias=False
                                  ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Reshape((4, 4, self.dims * 8))
        ])

        self.upsampling = tf.keras.Sequential([
            # 4
            UpsamplingConv2D(self.dims * 4),
            # 8
            UpsamplingConv2D(self.dims * 2),
            # 16
            UpsamplingConv2D(self.dims),
            # 32
            UpsamplingConv2D(self.dims // 2),
            # 64
            UpsamplingConv2D(self.dims // 4),
            # 128
            tf.keras.layers.Conv2DTranspose(filters=3,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding='valid',
                                            activation='tanh',
                                            kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                            )
            # 128
        ])

    def call(self, inputs, **kwargs):
        return self.upsampling(self.latent_projection(inputs))


class DownsamplingConv2D(tf.keras.layers.Layer):
    def __init__(self, n_filters):
        super(DownsamplingConv2D, self).__init__()
        self.n_filters = n_filters

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=self.n_filters,
                                   kernel_size=(5, 5),
                                   strides=(2, 2),
                                   padding='same',
                                   activation='linear',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02),
                                   use_bias=False
                                   ),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.LeakyReLU(.2)
        ])

    def call(self, inputs, **kwargs):
        return self.forward(inputs)


class Critic(tf.keras.layers.Layer):
    '''
    Only replaced batch normalization to layer normalization
    '''

    def __init__(self, dims=128):
        super(Critic, self).__init__()
        self.dims = dims

        self.downsampling = tf.keras.Sequential([
            # 128
            tf.keras.layers.Conv2D(filters=self.dims // 2,
                                   kernel_size=(5, 5),
                                   strides=(2, 2),
                                   padding='same',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                   ),
            tf.keras.layers.LeakyReLU(.2),
            # 64
            DownsamplingConv2D(self.dims),
            # 32
            DownsamplingConv2D(self.dims * 2),
            # 16
            DownsamplingConv2D(self.dims * 4),
            # 8
            DownsamplingConv2D(self.dims * 8),
            # 4
            tf.keras.layers.Conv2D(filters=1,
                                   kernel_size=(4, 4),
                                   padding='valid',
                                   activation='linear',
                                   kernel_initializer=tf.keras.initializers.random_normal(stddev=.02)
                                   ),
            # 1
            tf.keras.layers.Flatten()
        ])

    def call(self, inputs, **kwargs):
        return self.downsampling(inputs)


class WganGp(tf.keras.models.Model):
    '''
    Wasserstein GAN - Gradient Penalty
    '''
    def __init__(self,
                 lamb=10.,
                 n_critic=5,
                 alpha=.0001,
                 beta_1=.5,
                 beta_2=.9,
                 dim_latent=256,
                 batch_size=32
                 ):
        super(WganGp, self).__init__()
        self.lamb = lamb
        self.n_critic = n_critic
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.dim_latent = dim_latent
        self.batch_size = batch_size

        self.Generator = Generator()
        self.Generator.build((None, self.dim_latent))
        self.Critic = Critic()
        self.Critic.build((None, 128, 128, 3))
        self.compile()
        self.hist = []

    def compile(self):
        super(WganGp, self).compile()
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha,
                                                    beta_1=self.beta_1,
                                                    beta_2=self.beta_2
                                                    )
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha,
                                                    beta_1=self.beta_1,
                                                    beta_2=self.beta_2
                                                    )

    @tf.function
    def compute_x_hat(self, x, x_tilde):
        epsilon = tf.random.uniform(minval=0.,
                                    maxval=1.,
                                    shape=(self.batch_size, 1, 1, 1)
                                    )
        return epsilon * x + (1 - epsilon) * x_tilde

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
    def compute_gp(self, x_hat):
        avg_logit = self.Critic(x_hat, training=True)
        grads = tf.keras.backend.gradients(avg_logit, [x_hat])[0]
        l2norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean(tf.square(l2norm - 1.0))
        return gp

    @tf.function
    def train_step(self, data):
        img = tf.split(data,
                       axis=0,
                       num_or_size_splits=self.n_critic
                       )

        # 1. Update Critic for self.n_critic times
        mean_criticism_loss = 0.
        for i in range(self.n_critic):
            x = img[i]
            with tf.GradientTape() as tape:
                x_tilde = self.Generator(tf.random.normal((tf.shape(x)[0], self.dim_latent)))
                x_hat = self.compute_x_hat(x, x_tilde)
                gp = self.compute_gp(x_hat)
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
            x_tilde = self.Generator(tf.random.normal((tf.shape(x)[0], self.dim_latent)),
                                     training=True
                                     )
            loss = self.compute_gen_loss(self.Critic(x_tilde))
        grads = tape.gradient(loss, self.Generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(grads, self.Generator.trainable_variables)
        )

        return {'mean_criticism_loss': mean_criticism_loss, 'generation_loss': loss}
