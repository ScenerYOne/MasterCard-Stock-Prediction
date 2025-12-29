# 📈 MasterCard Stock Price Prediction with LSTM

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![LSTM](https://img.shields.io/badge/Model-LSTM-9B59B6)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?logo=numpy&logoColor=white)

A **deep learning-based stock price prediction system** using **LSTM (Long Short-Term Memory)** neural networks to forecast MasterCard stock prices. This project demonstrates time-series forecasting capabilities with automatic model training, evaluation, and result visualization.

---

## 🎯 Project Overview

This project implements an **LSTM-based time-series prediction model** to predict MasterCard stock prices. The system:

- **Trains on historical data** (2016-2020) to learn price patterns
- **Predicts future stock prices** using sequence-to-sequence learning
- **Evaluates model performance** with RMSE and MSE metrics
- **Generates visualizations** comparing actual vs predicted prices
- **Saves results** in timestamped directories for easy tracking

### 🧠 Core Technology

- **LSTM Neural Networks:** Captures long-term dependencies in time-series data
- **Sequence Learning:** Uses 60-day windows to predict next-day prices
- **MinMax Scaling:** Normalizes data for optimal neural network performance
- **Early Stopping:** Prevents overfitting during training

---

## ✨ Key Features

### 📊 Data Processing
- **Automatic Data Loading:** Reads MasterCard stock history from CSV
- **Time-Series Splitting:** Creates sequences of 60 time steps for prediction
- **Data Normalization:** MinMax scaling for improved model convergence

### 🤖 Model Architecture
- **LSTM Layer:** 256 units with tanh activation for sequence learning
- **Dense Output Layer:** Single neuron for price prediction
- **RMSprop Optimizer:** Efficient gradient-based optimization
- **MSE Loss Function:** Mean Squared Error for regression tasks

### 📈 Training & Evaluation
- **Flexible Training:** Train new models or load existing ones
- **Early Stopping:** Monitors loss with patience of 20 epochs
- **Performance Metrics:** Calculates RMSE and MSE on test data
- **Loss Visualization:** Plots training loss progression

### 💾 Results Management
- **Timestamped Runs:** Each execution creates a unique result directory
- **Multiple Formats:** Saves models in both `.h5` and `.keras` formats
- **Detailed Reports:** Generates comprehensive results text files
- **CSV Exports:** Saves predictions with actual vs predicted comparisons
- **Visual Plots:** Displays prediction graphs and loss curves

---

## 📂 Project Structure

```text
MasterCard Stock Prediction/
├── data/
│   └── Mastercard_stock_history.csv    # Historical stock price data
├── result/                              # Output directory
│   └── run_YYYYMMDD_HHMMSS/            # Timestamped result folders
│       ├── model.h5                     # Model in HDF5 format
│       ├── model.keras                  # Model in Keras 3.0+ format
│       ├── results.txt                  # Evaluation metrics and details
│       └── predictions.csv              # Actual vs predicted prices
├── main.py                              # Main execution script
├── model.py                             # LSTM/GRU model definitions
├── utils.py                             # Helper functions (plotting, data splitting)
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.9+**
- **pip** or **conda** package manager

### Step 1: Clone or Download Project

```bash
# Navigate to project directory
cd "MasterCard Stock Prediction"
```

### Step 2: Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using conda (recommended for TensorFlow)
conda create -n stock_prediction python=3.9
conda activate stock_prediction
pip install -r requirements.txt
```

### Step 3: Verify Data File

Ensure `data/Mastercard_stock_history.csv` exists with the following columns:
- `Date` (index column)
- `High` (target variable for prediction)
- `Open`, `Low`, `Close`, `Volume`
- `Dividends`, `Stock Splits` (optional, will be dropped)

---

## 💻 Usage

### Basic Execution

Simply run the main script:

```bash
python main.py
```

### Configuration Options

Edit `main.py` to customize:

```python
# Training period
tstart, tend = 2016, 2020  # Train on 2016-2020 data

# Test period
# Automatically uses data after 2020

# Model parameters
N_STEPS = 60              # Number of time steps (days) to look back
FEATURES = 1              # Number of features (currently only 'High' price)

# Training control
FORCE_RETRAIN = True      # Set to False to use existing model if available
```

### Output Files

After execution, check the `result/run_YYYYMMDD_HHMMSS/` directory:

- **`model.h5`** - Trained model in HDF5 format (compatible with older TensorFlow)
- **`model.keras`** - Trained model in Keras 3.0+ format (modern format)
- **`results.txt`** - Detailed evaluation metrics and training information
- **`predictions.csv`** - CSV file with columns:
  - `Actual_Price` - Real stock prices from test period
  - `Predicted_Price` - Model predictions
  - `Error` - Prediction error (Actual - Predicted)
  - `Absolute_Error` - Absolute value of error

### Visualizations

The script automatically displays:
1. **Training Loss Plot** - Shows model convergence during training
2. **Prediction Comparison Plot** - Visualizes actual vs predicted prices

---

## 🔧 Model Architecture Details

### LSTM Model Structure

```
Input Layer:  (60, 1)  →  60 time steps, 1 feature
    ↓
LSTM Layer:   256 units, tanh activation
    ↓
Dense Layer:  1 unit (price prediction)
    ↓
Output:       Single price value
```

### Training Configuration

- **Optimizer:** RMSprop
- **Loss Function:** Mean Squared Error (MSE)
- **Epochs:** Up to 100 (with early stopping)
- **Batch Size:** 32
- **Early Stopping:** Patience = 20 epochs
- **Validation:** Monitors training loss

---

## 📊 Evaluation Metrics

The model performance is evaluated using:

- **RMSE (Root Mean Squared Error):** Measures average prediction error magnitude
- **MSE (Mean Squared Error):** Measures average squared prediction error

Lower values indicate better model performance.

---

## 🎨 Customization

### Using Different Models

The `model.py` file includes both LSTM and GRU architectures:

```python
# In main.py, change:
from model import build_lstm_model  # or build_gru_model

# Then use:
model = build_gru_model(N_STEPS, FEATURES)  # Instead of build_lstm_model
```

### Adjusting Model Parameters

Edit `model.py` to modify:

- **LSTM/GRU Units:** Change `units=256` to adjust model capacity
- **Activation Function:** Modify `activation="tanh"` to "relu", "sigmoid", etc.
- **Optimizer:** Change `optimizer="RMSprop"` to "adam", "sgd", etc.
- **Loss Function:** Modify `loss="mse"` to "mae" (Mean Absolute Error)

### Using Different Features

To predict using multiple features (e.g., Open, High, Low, Close):

1. Modify data loading in `main.py`:
```python
# Instead of just "High"
training_data = dataset.loc[f"{tstart}":f"{tend}", ["High", "Low", "Close"]].values
```

2. Update `FEATURES`:
```python
FEATURES = 3  # Number of features
```

3. Ensure input shape matches: `(n_steps, features)`

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "File not found" error
**Solution:** Ensure `data/Mastercard_stock_history.csv` exists in the correct location

#### 2. Memory errors during training
**Solution:** 
- Reduce `N_STEPS` (e.g., from 60 to 30)
- Reduce batch size in `main.py` (e.g., `batch_size=16`)

#### 3. Poor prediction accuracy
**Solution:**
- Increase training data period
- Adjust `N_STEPS` (try 30, 60, 90)
- Modify model architecture (more LSTM units)
- Tune hyperparameters (learning rate, batch size)

#### 4. TensorFlow/Keras version conflicts
**Solution:**
```bash
pip install --upgrade tensorflow
# Or use specific version
pip install tensorflow==2.13.0
```

---

## 📚 Dependencies

| Package | Purpose |
|:---|:---|
| **TensorFlow/Keras** | Deep learning framework for LSTM models |
| **NumPy** | Numerical computations and array operations |
| **Pandas** | Data manipulation and CSV reading |
| **Matplotlib** | Visualization and plotting |
| **Scikit-learn** | Data preprocessing (MinMaxScaler) |

---

## 📖 References

- **TensorFlow/Keras Documentation:** [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **LSTM Networks:** Understanding Long Short-Term Memory for time-series forecasting
- **Stock Market Data:** Historical MasterCard stock prices

---

**MasterCard Stock Prediction** - *Predicting the future, one day at a time.* 📈

