from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv1D, LSTM, Flatten, Input, Concatenate, LeakyReLU, Dropout
from config import *

def create_generator(noise_shape, condition_shape, output_sequence_length, num_features):
    """CNN Generator with Condition"""
    noise_input = Input(shape=noise_shape, name='gen_noise_input') 
    condition_input = Input(shape=condition_shape, name='gen_condition_input')
    
    # Process Condition (History) with 1D Conv to extract features
    # (Batch, Lookback, Feat) -> (Batch, New_Len, Filters)
    c = Conv1D(filters=32, kernel_size=3, padding='same')(condition_input)
    c = LeakyReLU(negative_slope=LEAKY_RELU_ALPHA)(c)
    c = Flatten()(c)
    
    n = Flatten()(noise_input)
    
    merged = Concatenate()([n, c])
    
    # Upsampling / Generation path
    x = Dense(output_sequence_length * CNN_LSTM_GAN_G_DENSE_UNITS)(merged)
    # Applying LeakyReLU after Dense upscale is good practice
    x = LeakyReLU(negative_slope=LEAKY_RELU_ALPHA)(x)
    x = Dropout(0.2)(x)
    
    x = Reshape((output_sequence_length, CNN_LSTM_GAN_G_DENSE_UNITS))(x)
    output = Conv1D(num_features, kernel_size=1, activation=ACTIVATION_TANH)(x)
    
    return Model([noise_input, condition_input], output, name='Generator')

def create_discriminator(input_shape):
    """LSTM Discriminator"""
    model = Sequential()
    model.add(LSTM(units=LSTM_GAN_D_UNITS, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(LSTM(units=LSTM_GAN_D_UNITS, return_sequences=False))
    model.add(Dropout(LSTM_DROPOUT))
    model.add(Dense(1)) # Linear
    return model

def create_model(input_shape_noise, input_shape_condition, input_shape_disc, output_sequence_length, num_features):
    """
    Creates GAN_CNN_LSTM model.
    """
    generator = create_generator(input_shape_noise, input_shape_condition, output_sequence_length, num_features)
    discriminator = create_discriminator(input_shape_disc)
    
    return None, generator, discriminator
