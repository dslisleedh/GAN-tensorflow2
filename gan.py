import tensorflow as tf
import numpy as np

class GAN(tf.keras.models.Model):
    def __init__(self, batch_size = 100, dim_latent = 25, disc_n_downsampling = 2, disc_n_filters = 128, disc_kernel_size = 3):
        super(GAN, self).__init__()
        self.batch_size = batch_size
        self.dim_latent = dim_latent
        self.disc_n_downsampling = disc_n_downsampling
        self.disc_n_filters = disc_n_filters
        self.disc_kernel_size = disc_kernel_size

        self.D = Discrimimator(self.disc_n_downsampling, self.disc_n_filters, self.disc_kernel_size)
        self.G = Generator(self.dim_latent)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, Input):
        X = tf.reshape(Input, shape = (-1, 28, 28, 1))
        X += tf.random.normal(stddev=.025, shape = tf.shape(X))
        random_latent = tf.random.normal(shape = (self.batch_size, self.dim_latent))

        generated_images = self.G(random_latent)
        combined_images = tf.concat([generated_images, X], axis = 0)
        y = tf.concat([tf.ones(shape = (self.batch_size, 1)),
                       tf.zeros(shape = (self.batch_size, 1)),],
                      axis = 0
                     )

        with tf.GradientTape() as tape:
            y_pred = self.D(combined_images)
            d_loss = self.loss_fn(y, y_pred)
        grads = tape.gradient(d_loss, self.D.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(grads, self.D.trainable_variables)
        )

        with tf.GradientTape() as tape:
            y_pred = self.D(self.G(random_latent))
            g_loss = self.loss_fn(tf.zeros(shape = (self.batch_size, 1)), y_pred)
        grads = tape.gradient(g_loss, self.G.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(grads, self.G.trainable_variables)
        )
        return {"d_loss" : d_loss, "g_loss" : g_loss}



class Discrimimator(tf.keras.layers.Layer):
    def __init__(self, n_downsampling, n_filters, kernel_size):
        super(Discrimimator, self).__init__()
        self.n_downsampling = n_downsampling
        self.n_filters = n_filters
        self.kernel_size = kernel_size

        self.Downsampling = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters = self.n_filters * (i+1),
                                   kernel_size = self.kernel_size,
                                   strides = (2,2),
                                   padding = 'same',
                                   activation = 'relu',
                                   kernel_initializer = 'he_normal'
                                   ) for i in range(self.n_downsampling)
        ])

        self.Classifier = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1,
                                  activation = 'sigmoid'
                                  )
        ])

    def call(self, X):
        y = self.Downsampling(X)
        y = self.Classifier(y)
        return y

class Generator(tf.keras.layers.Layer):
    def __init__(self, dim_latent):
        super(Generator, self).__init__()
        self.dim_latent = dim_latent

        self.FC = tf.keras.layers.Dense(7*7*self.dim_latent,
                                        activation = 'relu',
                                        kernel_initializer = 'he_normal'
                                       )
        self.Upsampling = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(64,
                                            kernel_size = (4,4),
                                            strides = (2,2),
                                            padding = 'same',
                                            activation = 'relu',
                                            kernel_initializer = 'he_normal'
                                            ),
            tf.keras.layers.Conv2DTranspose(128,
                                            kernel_size = (4,4),
                                            strides = (2,2),
                                            padding = 'same',
                                            activation = 'relu',
                                            kernel_initializer = 'he_normal'
                                            ),
            tf.keras.layers.Conv2D(1,
                                   kernel_size = (7,7),
                                   activation = 'sigmoid',
                                   padding = 'same'
                                   )
        ])

    def call(self, X):
        y = self.FC(X)
        y = tf.reshape(y, shape = (-1, 7, 7, self.dim_latent))
        y = self.Upsampling(y)
        return y