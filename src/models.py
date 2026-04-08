from tensorflow.keras import layers, Model, Input, optimizers
import config 

class ModelBuilder:
    def __init__(self, latent_dim=config.LATENT_DIM):
        self.latent_dim = latent_dim

    def build_autoencoder(self, input_shape=(config.MAX_SAMPLES, config.CHANNELS)):
        """Builds an Encoder-Decoder structure used for BOTH SAE and MAE"""
        # ENCODER
        encoder_input = Input(shape=input_shape)
        x = layers.Conv1D(32, 5, activation="relu", padding="same")(encoder_input)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 5, activation="relu", padding="same")(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, activation="relu", name="latent_features")(x)
        encoder = Model(encoder_input, latent, name="Encoder")

        # DECODER
        decoder_input = Input(shape=(self.latent_dim,))
        x = layers.Dense((input_shape[0]//4) * 64, activation="relu")(decoder_input)
        x = layers.Reshape((input_shape[0]//4, 64))(x)
        x = layers.UpSampling1D(2)(x)
        x = layers.Conv1D(32, 5, activation="relu", padding="same")(x)
        x = layers.UpSampling1D(2)(x)
        output = layers.Conv1D(config.CHANNELS, 5, activation="linear", padding="same")(x)
        
        decoder = Model(decoder_input, output, name="Decoder")
        
        # FULL AE
        autoencoder_output = decoder(encoder(encoder_input))
        autoencoder = Model(encoder_input, autoencoder_output, name="Autoencoder")
        autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mse")
        
        return autoencoder, encoder

    def build_bilstm(self, input_shape=(config.MAX_SAMPLES, config.CHANNELS), num_classes=5):
        """The 'High Accuracy' Deep Learning Mode"""
        inputs = Input(shape=input_shape)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_cnn(self, input_shape=(config.MAX_SAMPLES, config.CHANNELS), num_classes=5):
        """1D-CNN for ECG classification"""
        inputs = Input(shape=input_shape)
        x = layers.Conv1D(32, 7, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model