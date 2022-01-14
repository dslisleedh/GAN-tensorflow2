import tensorflow as tf

class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()


class Transfer(tf.keras.layers.Layer):
    def __init__(self):
        super(Transfer, self).__init__()


class Cyclegan(tf.keras.models.Mode):
    def __init__(self, lamb):
        super(Cyclegan, self).__init__()
        self.lamb = lamb

        self.F = Transfer()
        self.G = Transfer()
        self.Disc_x = Discriminator()
        self.Disc_y = Discriminator()

    def compile(self, disc_x_optimizer,
                disc_y_optimizer,
                g_optimizer,
                f_optimizer
                ):
        self.disc_x_optimizer = disc_x_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        self.g_optimizer = g_optimizer
        self.f_optimizer = f_optimizer

    @tf.function
    def train_step(self, data):
        X, y = data

        #1 update Disc_y
        with tf.GradientTape() as tape:
            y_hat = self.G(X)
            disc_y_loss = tf.reduce_mean(
                tf.square(self.Disc_y(y) - 1.)
            ) + tf.reduce_mean(
                tf.square(self.Disc_y(y_hat))
            )
        grads = tape.gradient(disc_y_loss, self.Disc_y.trainable_variables)
        self.disc_x_optimizer.apply_gradients(
            zip(grads, self.Disc_y.trainable_variables)
        )

        #2 update Disc_x
        with tf.GradientTape() as tape:
            X_hat = self.F(y)
            disc_x_loss = tf.reduce_mean(
                tf.square(self.Disc_x(X) - 1.)
            ) + tf.reduce_mean(
                tf.square(self.Disc_x(X_hat))
            )
        grads = tape.gradient(disc_x_loss, self.Disc_x.trainable_variables)
        self.disc_y_optimizer.apply_gradients(
            zip(grads, self.Disc_x.trainable_variables)
        )

        #3 update G,F by X
        with tf.GradientTape() as tape:
            y_hat = self.G(X)
            X_recon = self.F(y_hat)

            gen_loss_x = tf.reduce_mean(
                tf.square(self.Disc_y(y_hat) - 1.)
            )
            cycle_loss_x = tf.reduce_mean(
                tf.abs(X_recon - X)
            )
            loss_x = gen_loss_x + self.lamb * cycle_loss_x
        grads = tape.gradient(loss_x, self.G.trainable_variables + self.F.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(grads, self.G.trainable_variables + self.F.trainable_variables)
        )

        #4 update F,G by y
        with tf.GradientTape() as tape:
            X_hat = self.F(y)
            y_recon = self.G(X_recon)

            gen_loss_y = tf.reduce_mean(
                tf.square(self.Disc_x(X_hat) - 1.)
            )
            cycle_loss_y = tf.reduce_mean(
                tf.abs(y_recon - y)
            )
            loss_y = gen_loss_y + self.lamb * cycle_loss_y
        grads = tape.gradient(loss_y, self.F.trainable_variables + self.G.trainable_variables)
        self.f_optimizer.apply_gradients(
            zip(grads, self.F.trainable_variables + self.G.trainable_variables)
        )

        return {'Disc. X_loss' : disc_x_loss,
                'Disc. y_loss' : disc_y_loss,
                'Generation_loss' : gen_loss_x + gen_loss_y,
                'Cycle_loss' : cycle_loss_x + cycle_loss_y
                }