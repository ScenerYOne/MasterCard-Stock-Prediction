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

def plot_predictions(test_set, predicted_price, title="Stock Price Prediction"):
    """ฟังก์ชันสำหรับวาดกราฟเปรียบเทียบราคาจริงกับราคาทำนาย"""
    plt.figure(figsize=(16, 4))
    plt.plot(test_set, color="gray", label="Real Price")
    plt.plot(predicted_price, color="red", label="Predicted Price")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()