import numpy as np
import matplotlib.pyplot as plt

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def plot_predictions(test_set, predicted_price, title="MasterCard Stock Price Prediction"):
    plt.figure(figsize=(14, 5))
    plt.plot(test_set, color="gray", label="Real Price")
    plt.plot(predicted_price, color="red", label="Predicted Price")
    plt.title(title)
    plt.xlabel("Time (Days)")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()