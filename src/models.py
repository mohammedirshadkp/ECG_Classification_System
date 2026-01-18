import tensorflow as tf
from tensorflow.keras import layers, Model, Input, optimizers
import numpy as np

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker]

    def train_step(self, data):
        if isinstance(data, tuple): data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # 1. Reconstruction Loss with clipping
            reconstruction = tf.clip_by_value(reconstruction, 1e-7, 1.0 - 1e-7)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.keras.losses.mse(data, reconstruction), axis=-1)
            )
            
            # 2. KL Divergence with clipping to prevent NaNs
            z_log_var = tf.clip_by_value(z_log_var, -10, 10) 
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        # 3. Gradient Clipping - prevents the math from exploding
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in grads]
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"loss": self.total_loss_tracker.result()}
    
class ModelBuilder:
    def __init__(self, latent_dim=32):
        self.latent_dim = latent_dim
        # Learning rate and clipnorm for stability
        self.opt = optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)

    def build_ae(self, input_dim=1000):
        encoder_input = Input(shape=(input_dim,))
        x = layers.Dense(256, activation="relu")(encoder_input)
        latent = layers.Dense(self.latent_dim, activation="relu")(x)
        encoder = Model(encoder_input, latent, name="AE_Encoder")

        decoder_input = Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation="relu")(decoder_input)
        output = layers.Dense(input_dim, activation="sigmoid")(x)
        decoder = Model(decoder_input, output, name="AE_Decoder")

        autoencoder = Model(encoder_input, decoder(latent), name="AE")
        autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss="mse")
        return autoencoder, encoder

    def build_vae(self, input_dim=1000):
        # Encoder
        encoder_input = Input(shape=(input_dim,))
        x = layers.Dense(256, activation="relu")(encoder_input)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)

        def sampling(args):
            z_m, z_lv = args
            epsilon = tf.random.normal(shape=tf.shape(z_m))
            return z_m + tf.exp(0.5 * z_lv) * epsilon

        z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])
        encoder = Model(encoder_input, [z_mean, z_log_var, z], name="VAE_Encoder")

        # Decoder
        decoder_input = Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation="relu")(decoder_input)
        output = layers.Dense(input_dim, activation="sigmoid")(x)
        decoder = Model(decoder_input, output, name="VAE_Decoder")

        vae = VAE(encoder, decoder)
        
        # --- FIX HERE: Added run_eagerly=True ---
        vae.compile(optimizer=optimizers.Adam(learning_rate=0.0001), run_eagerly=True)
        return vae, encoder

    def build_cnn(self, input_length):
        model = tf.keras.Sequential([
            layers.Input(shape=(input_length, 1)),
            layers.Conv1D(32, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        return model