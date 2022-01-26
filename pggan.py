import tensorflow as tf
import numpy as np


class PixelNormalization(tf.keras.layers.Layer):
    def __init__(self, eps=1e-8):
        super(PixelNormalization, self).__init__()
        self.eps = eps

    def call(self, inputs, **kwargs):
        return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True) + self.eps)


class Pggan(tf.keras.models.Model):
    def __init__(self,
                 build_datasets,
                 output_channel=3,
                 output_size=512,
                 dim_latent=100
                 ):
        super(Pggan, self).__init__()
        self.output_channel = output_channel
        n_upsampling = (np.log(output_size) / np.log(2)) - 1
        if n_upsampling % 1 != 0:
            assert ValueError('size must power of 2')
        else:
            self.output_size = output_size
            self.n_layers = n_upsampling
        self.dim_latent = dim_latent

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