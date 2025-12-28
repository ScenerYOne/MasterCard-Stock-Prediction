import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from model import build_lstm_model
from utils import split_sequence, plot_predictions

# 1. Load Data
dataset = pd.read_csv("data\Mastercard_stock_history.csv", index_col="Date", parse_dates=["Date"])
dataset = dataset.drop(["Dividends", "Stock Splits"], axis=1, errors='ignore')

# 2. Split Train/Test (2016-2020 สำหรับเทรน, 2021 สำหรับทดสอบ)
tstart, tend = 2016, 2020
training_set = dataset.loc[f"{tstart}":f"{tend}", "High"].values.reshape(-1, 1)
test_set = dataset.loc[f"{tend+1}":, "High"].values

# 3. Scale Data
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# 4. Prepare Sequences
n_steps = 60
features = 1
X_train, y_train = split_sequence(training_set_scaled, n_steps)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)

# 5. Build and Train Model
model = build_lstm_model(n_steps, features)
print("Start Training...")
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)

# 6. Prediction
dataset_total = dataset["High"]
inputs = dataset_total[len(dataset_total) - len(test_set) - n_steps:].values.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test, _ = split_sequence(inputs, n_steps)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# 7. Evaluate and Visualization
rmse = np.sqrt(mean_squared_error(test_set, predicted_stock_price))
print(f"RMSE: {rmse:.2f}")
plot_predictions(test_set, predicted_stock_price, "MasterCard Stock Price Prediction (LSTM)")