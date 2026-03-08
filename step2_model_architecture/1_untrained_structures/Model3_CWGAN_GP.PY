from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Dropout, Input, Concatenate, LeakyReLU
from config import *

def create_generator(noise_shape, condition_shape, output_sequence_length, num_features):
    """Dense Generator with Condition"""
    # Inputs
    noise_input = Input(shape=noise_shape, name='gen_noise_input')
    condition_input = Input(shape=condition_shape, name='gen_condition_input')
    
    # Flatten inputs
    noise_flat = Flatten()(noise_input)
    condition_flat = Flatten()(condition_input)
    
    # Concatenate noise and condition
    merged = Concatenate()([noise_flat, condition_flat])
    
    # Hidden Layers
    x = Dense(128)(merged)
    x = LeakyReLU(negative_slope=LEAKY_RELU_ALPHA)(x)
    
    x = Dense(256)(x)
    x = LeakyReLU(negative_slope=LEAKY_RELU_ALPHA)(x)
    
    x = Dense(output_sequence_length * num_features, activation='tanh')(x)
    
    # Output
    output = Reshape((output_sequence_length, num_features))(x)
    
    return Model([noise_input, condition_input], output, name='Generator')

def create_discriminator(input_shape):
    """Dense Discriminator"""
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    
    model.add(Dense(GAN_DENSE_D_UNITS))
    model.add(LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
    model.add(Dropout(GAN_DENSE_D_DROPOUT))
    
    model.add(Dense(GAN_DENSE_D_UNITS // 2))
    model.add(LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
    model.add(Dropout(GAN_DENSE_D_DROPOUT))
    
    model.add(Dense(1)) # Linear output for WGAN
    return model

def create_model(input_shape_noise, input_shape_condition, input_shape_disc, output_sequence_length, num_features):
    """
    Creates GAN_Dense_Dense model.
    """
    generator = create_generator(input_shape_noise, input_shape_condition, output_sequence_length, num_features)
    discriminator = create_discriminator(input_shape_disc)
    
    return None, generator, discriminator
