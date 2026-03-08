from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, LeakyReLU
from config import *

def create_cnn_lstm_model(input_shape):
    """
    Creates a CNN-LSTM model for stock price prediction.
    """
    model = Sequential()

    # CNN layers
    # Remove activation='relu' and add LeakyReLU layer
    model.add(Conv1D(filters=CNN_FILTERS, kernel_size=CNN_KERNEL_SIZE, input_shape=input_shape))
    model.add(LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
    
    model.add(MaxPooling1D(pool_size=CNN_POOL_SIZE))

    # LSTM layers
    # Using default tanh for LSTM stability
    model.add(LSTM(units=CNN_LSTM_UNITS, return_sequences=True))
    model.add(Dropout(CNN_LSTM_DROPOUT))
    model.add(LSTM(units=CNN_LSTM_UNITS, return_sequences=False))
    model.add(Dropout(CNN_LSTM_DROPOUT))

    # Dense output layer
    model.add(Dense(units=CNN_DENSE_UNITS))
    model.add(LeakyReLU(negative_slope=LEAKY_RELU_ALPHA))
    
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer=OPTIMIZER_ADAM, loss=LOSS_MSE)

    return model
