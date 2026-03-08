from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Input, Concatenate, RepeatVector, Flatten, Dropout
from config import *

def create_generator(noise_shape, condition_shape, output_sequence_length, num_features):
    """LSTM Generator with Condition"""
    # noise_shape: (Latent_dim,)
    # condition_shape: (Lookback, Features)
    
    # Input: Noise (Latent Vector)
    noise_input = Input(shape=noise_shape, name='gen_noise_input') 
    # Input: Condition (History)
    condition_input = Input(shape=condition_shape, name='gen_condition_input')
    
    # Process Condition with LSTM to extract features
    cond_encoded = LSTM(units=50, return_sequences=False)(condition_input)
    # Add Dropout for robustness
    cond_encoded = Dropout(LSTM_DROPOUT)(cond_encoded)
    
    # Process Noise
    noise_flat = Flatten()(noise_input)
    
    # Merge
    merged = Concatenate()([noise_flat, cond_encoded])
    
    # Repeat to match sequence length for generation
    x = RepeatVector(output_sequence_length)(merged)
    
    # LSTM Decoder
    x = LSTM(units=LSTM_GAN_G_UNITS, return_sequences=True)(x)
    x = Dropout(LSTM_DROPOUT)(x)
    x = LSTM(units=LSTM_GAN_G_UNITS, return_sequences=True)(x)
    x = Dropout(LSTM_DROPOUT)(x)
    output = TimeDistributed(Dense(num_features, activation='tanh'))(x)
    
    return Model([noise_input, condition_input], output, name='Generator')

def create_discriminator(input_shape):
    """LSTM Discriminator"""
    model = Sequential()
    model.add(LSTM(units=LSTM_GAN_D_UNITS, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(LSTM(units=LSTM_GAN_D_UNITS, return_sequences=False))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(Dense(1)) # Linear output for WGAN
    return model

def create_model(input_shape_noise, input_shape_condition, input_shape_disc, output_sequence_length, num_features):
    """
    Creates GAN_LSTM_LSTM model.
    """
    generator = create_generator(input_shape_noise, input_shape_condition, output_sequence_length, num_features)
    discriminator = create_discriminator(input_shape_disc)
    
    return None, generator, discriminator
