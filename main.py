import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

from model import build_lstm_model
from utils import split_sequence, plot_predictions, plot_loss

FILE_PATH = "data\Mastercard_stock_history.csv"
MODEL_NAME = "mastercard_lstm_model.h5"
RESULT_DIR = "result"
N_STEPS = 60
FEATURES = 1
FORCE_RETRAIN = True  


os.makedirs(RESULT_DIR, exist_ok=True)

if not os.path.exists(FILE_PATH):
    print(f"Error: ค้นหาไฟล์ {FILE_PATH} ไม่พบ!")
    exit()

dataset = pd.read_csv(FILE_PATH, index_col="Date", parse_dates=["Date"])
dataset = dataset.drop(["Dividends", "Stock Splits"], axis=1, errors='ignore')


tstart, tend = 2016, 2020
training_data = dataset.loc[f"{tstart}":f"{tend}", "High"].values.reshape(-1, 1)
test_data = dataset.loc[f"{tend+1}":, "High"].values.reshape(-1, 1)

sc = MinMaxScaler(feature_range=(0, 1))
training_scaled = sc.fit_transform(training_data)

X_train, y_train = split_sequence(training_scaled, N_STEPS)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], FEATURES)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# สร้างโฟลเดอร์ย่อยสำหรับรันนี้
run_dir = os.path.join(RESULT_DIR, f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)
print(f" Results will be saved to: {run_dir} \n")

model_trained = False
history = None

if os.path.exists(MODEL_NAME) and not FORCE_RETRAIN:
    print(f" Loading existing model: {MODEL_NAME} ")
    try:
        model = load_model(MODEL_NAME)
    except ValueError as e:
        print(f"Warning: Error loading model with compile=True: {e}")
        print("Attempting to load model without compilation")
        model = load_model(MODEL_NAME, compile=False)
        model.compile(optimizer="RMSprop", loss="mse")
    print(" Using existing model for prediction ")
else:
    if FORCE_RETRAIN and os.path.exists(MODEL_NAME):
        print(" FORCE_RETRAIN is True: Training new model (overwriting existing) ")
    else:
        print(" Training new model ")
    
    model = build_lstm_model(N_STEPS, FEATURES)
    
    early_stop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        callbacks=[early_stop], 
        verbose=1
    )
    
    model_trained = True
    model.save(MODEL_NAME)
    print(f" Model saved as {MODEL_NAME} ")
    
    if history:
        plot_loss(history)


model_h5_path = os.path.join(run_dir, "model.h5")
model_keras_path = os.path.join(run_dir, "model.keras")

model.save(model_h5_path)
model.save(model_keras_path)
print(f" Model saved to {run_dir}/ (model.h5 and model.keras) ")

dataset_total = dataset["High"].values.reshape(-1, 1)
inputs = dataset_total[len(dataset_total) - len(test_data) - N_STEPS:]
inputs_scaled = sc.transform(inputs)

X_test, _ = split_sequence(inputs_scaled, N_STEPS)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], FEATURES)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

rmse = np.sqrt(mean_squared_error(test_data, predicted_stock_price))
print(f"Root Mean Squared Error: {rmse:.2f}")


results_path = os.path.join(run_dir, "results.txt")

with open(results_path, 'w', encoding='utf-8') as f:
    f.write("=== MasterCard Stock Prediction Model Results ===\n\n")
    f.write(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Run Directory: {run_dir}\n\n")
    f.write(f"Model: LSTM\n")
    f.write(f"Model Status: {'Newly Trained' if model_trained else 'Loaded from Existing'}\n")
    f.write(f"Training Period: {tstart}-{tend}\n")
    f.write(f"Test Period: {tend+1}-\n")
    f.write(f"N_STEPS: {N_STEPS}\n")
    f.write(f"Features: {FEATURES}\n\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    f.write(f"Mean Squared Error (MSE): {mean_squared_error(test_data, predicted_stock_price):.4f}\n")
    f.write(f"\nTest Data Points: {len(test_data)}\n")
    f.write(f"Predicted Data Points: {len(predicted_stock_price)}\n")
    if history:
        f.write(f"\nTraining History:\n")
        f.write(f"  Final Loss: {history.history['loss'][-1]:.6f}\n")
        f.write(f"  Total Epochs: {len(history.history['loss'])}\n")
        f.write(f"  Best Loss: {min(history.history['loss']):.6f}\n")

print(f" Results saved to {results_path} ")

predictions_df = pd.DataFrame({
    'Actual_Price': test_data.flatten(),
    'Predicted_Price': predicted_stock_price.flatten(),
    'Error': (test_data.flatten() - predicted_stock_price.flatten()),
    'Absolute_Error': np.abs(test_data.flatten() - predicted_stock_price.flatten())
})
predictions_csv_path = os.path.join(run_dir, "predictions.csv")
predictions_df.to_csv(predictions_csv_path, index=False)
print(f" Predictions saved to {predictions_csv_path} ")

print(f"\n=== All files saved to: {run_dir} ===")
print(f"  - model.h5")
print(f"  - model.keras")
print(f"  - results.txt")
print(f"  - predictions.csv")

plot_predictions(test_data, predicted_stock_price)