from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def build_lstm_model(n_steps, features):
    model = Sequential()
    model.add(LSTM(units=125, activation="tanh", input_shape=(n_steps, features)))
    model.add(Dense(units=1))
    model.compile(optimizer="RMSprop", loss="mse")
    return model