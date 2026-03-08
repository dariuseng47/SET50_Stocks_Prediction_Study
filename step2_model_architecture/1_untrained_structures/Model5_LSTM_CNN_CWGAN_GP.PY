from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Input, Concatenate, RepeatVector, LeakyReLU, Dropout
from config import *

def create_generator(noise_shape, condition_shape, output_sequence_length, num_features):
    """LSTM Generator with Condition"""
    # Identical structure to ALT1 Generator
    noise_input = Input(shape=noise_shape, name='gen_noise_input') 
    condition_input = Input(shape=condition_shape, name='gen_condition_input')
    
    cond_encoded = LSTM(units=50, return_sequences=False)(condition_input)
    cond_encoded = Dropout(LSTM_DROPOUT)(cond_encoded)
    
    noise_flat = Flatten()(noise_input)
    
    merged = Concatenate()([noise_flat, cond_encoded])
    
    x = RepeatVector(output_sequence_length)(merged)
    
    x = LSTM(units=LSTM_GAN_G_UNITS, return_sequences=True)(x)
    x = Dropout(LSTM_DROPOUT)(x)
    x = LSTM(units=LSTM_GAN_G_UNITS, return_sequences=True)(x)
    x = Dropout(LSTM_DROPOUT)(x)
    output = TimeDistributed(Dense(num_features, activation='tanh'))(x)
    
    return Model([noise_input, condition_input], output, name='Generator')

def create_discriminator(input_shape):
    """CNN Discriminator"""
    model = Sequential()
    # Remove activation='relu' and add LeakyReLU layer
    model.add(Conv1D(filters=LSTM_CNN_GAN_D_FILTERS, kernel_size=LSTM_CNN_GAN_D_KERNEL_SIZE, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
    
    # Check if pooling is valid for input length
    # If input length is 1, pool size 2 will fail?
    # Actually, padding='same' in Conv1D keeps it 1.
    # MaxPooling1D with pool_size=2 on length 1 might fail or reduce to 1.
    # Safe to use padding='same' for Conv1D. 
    # For pooling, if length < pool_size, it might be an issue.
    # If length is 1, we should probably skip pooling or use pool_size=1.
    if input_shape[0] > 1:
        model.add(MaxPooling1D(pool_size=LSTM_CNN_GAN_D_POOL_SIZE))
    
    model.add(Flatten())
    model.add(Dropout(GAN_DENSE_D_DROPOUT))
    model.add(Dense(1)) # Linear output for WGAN
    return model

def create_model(input_shape_noise, input_shape_condition, input_shape_disc, output_sequence_length, num_features):
    """
    Creates GAN_LSTM_CNN model.
    """
    generator = create_generator(input_shape_noise, input_shape_condition, output_sequence_length, num_features)
    discriminator = create_discriminator(input_shape_disc)
    
    return None, generator, discriminator
