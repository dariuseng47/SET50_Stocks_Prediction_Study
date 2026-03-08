from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LeakyReLU
from config import *

def create_lstm_model(input_shape):
    """
    Creates a Long Short-Term Memory (LSTM) model matching the user's specific architecture.
    Structure: Input -> LSTM -> Dropout -> LSTM -> Dropout -> Dense -> Output
    """
    model = Sequential([
        Input(shape=input_shape),
        # LSTM default activation is tanh. Using that is safer than ReLU for gradients in RNNs.
        # If we strictly want LeakyReLU on the output of LSTM, we can do activation=None + LeakyReLU layer,
        # but internal gates still use sigmoid/tanh. 
        # Here we replace the explicit 'relu' with default (tanh) for stability, 
        # or we can try to pass LeakyReLU if supported. 
        # Standard practice: Use tanh for LSTM, LeakyReLU for Dense/Conv.
        LSTM(units=LSTM_UNITS, return_sequences=True), 
        # Note: Previous was activation='relu'. Removed to use default tanh.
        
        Dropout(LSTM_DROPOUT),
        
        LSTM(units=LSTM_UNITS),
        # Note: Previous was activation='relu'. Removed to use default tanh.
        
        Dropout(LSTM_DROPOUT),
        
        Dense(units=LSTM_DENSE_UNITS), # No activation here
        LeakyReLU(negative_slope=LEAKY_RELU_ALPHA), # Apply LeakyReLU separately
        
        Dense(units=1)
    ])

    # Compile the model
    model.compile(optimizer=OPTIMIZER_ADAM, loss=LOSS_MSE)

    return model
