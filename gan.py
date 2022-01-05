import tensorflow as tf
import numpy as np

class GAN(tf.keras.models.Model):
    def __init__(self, dim_latent = 25):
        super(GAN, self).__init__()
        self.dim_latent = dim_latent

        self.D = Discrimimator()
        self.G = Generator(self.dim_latent)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, X):
        #1 Update Discriminator
        generated_images = self.G(tf.random.normal(shape = (tf.shape(X)[0], self.dim_latent)))
        with tf.GradientTape() as tape:
            output_true = self.D(X)
            output_fake = self.D(generated_images)
            t_loss = self.loss_fn(tf.zeros(shape = tf.shape(output_true)), output_true)
            f_loss = self.loss_fn(tf.ones(shape = tf.shape(output_fake)), output_fake)
            d_loss = t_loss + f_loss
        grads = tape.gradient(d_loss, self.D.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(grads, self.D.trainable_variables)
        )

        #2 Update Generator
        with tf.GradientTape() as tape:
            y_pred = self.D(self.G(tf.random.normal(shape = (tf.shape(X)[0], self.dim_latent))))
            g_loss = self.loss_fn(tf.zeros(shape = tf.shape(y_pred)), y_pred)
        grads = tape.gradient(g_loss, self.G.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(grads, self.G.trainable_variables)
        )
        return {"d_loss" : d_loss, "g_loss" : g_loss}


class Discrimimator(tf.keras.layers.Layer):
    def __init__(self):
        super(Discrimimator, self).__init__()

        self.Downsampling = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters =64,
                                   kernel_size = (3,3),
                                   strides = (2,2),
                                   padding = 'same',
                                   kernel_initializer = 'he_normal'
                                   ),
            tf.keras.layers.LeakyReLU(.15),
            tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=(3, 3),
                                   strides=(2, 2),
                                   padding='same',
                                   kernel_initializer='he_normal'
                                   ),
            tf.keras.layers.LeakyReLU(.15),
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
            tf.keras.layers.Conv2DTranspose(128,
                                            kernel_size = (4,4),
                                            strides = (2,2),
                                            padding = 'same',
                                            activation = 'relu',
                                            kernel_initializer = 'he_normal'
                                            ),
            tf.keras.layers.Conv2DTranspose(64,
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