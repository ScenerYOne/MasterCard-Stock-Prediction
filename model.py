from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout

def build_lstm_model(n_steps, features):
    """โมเดล LSTM"""
    model = Sequential()
    model.add(LSTM(units=256, activation="tanh", input_shape=(n_steps, features)))
    model.add(Dense(units=1))
    model.compile(optimizer="RMSprop", loss="mse")
    return model

def build_gru_model(n_steps, features):
    """โมเดล GRU """
    model = Sequential()
    model.add(GRU(units=256, activation="tanh", input_shape=(n_steps, features)))
    model.add(Dense(units=1))
    model.compile(optimizer="RMSprop", loss="mse")
    return model