import tensorflow as tf


class Generator(tf.keras.layers.Layer):
    def __init__(self):
        super(Generator, self).__init__()


class Critic(tf.keras.layers.Layer):
    def __init__(self):
        super(Critic, self).__init__()


class Wgan(tf.keras.models.Model):
    def __init__(self):
        super(Wgan, self).__init__()
        self.alpha = .00005
        self.c = 0.01
        self.n_critic = 5

        self.Generator = Generator()
        self.Generator.build()
        self.Critic = Critic()
        self.Discriminator.build()
        self.compile()

    def compile(self):
        self.g_optimizer = tf.keras.optimizers.RMSprop(self.alpha)
        self.c_optimizer = tf.keras.optimizers.RMSprop(self.alpha)

    @tf.function
    def compute_wasserstein(self, label, pred):
        '''
        set label as

        true_label : 1
        fake_label : -1

        to make same with ${ mean(true) - mean(pred) }#
        '''
        loss = tf.reduce_mean(
            label * pred
        )
        return loss

    @tf.function
    def train_step(self, img):
        true_label = tf.ones(tf.shape(img)[0])
        fake_label = -tf.ones(tf.shape(img)[0])

        for _ in range(self.n_critic):
            fake_img = self.Generator(tf.random.normal((tf.shape(img)[0], 100)))
            with tf.GradientTape() as tape:
                true_logit = self.Critic(img)
                fake_logit = self.Critic(fake_img)

                true_loss = self.compute_wasserstein(true_label, true_logit)
                fake_loss = self.Critic(fake_label, fake_logit)
                loss = true_loss + fake_loss
            grads = tape.gradient(loss, self.Critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(grads, self.Critic.trainable_variables)
            )
