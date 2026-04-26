# ⚡ Electric Production Forecasting — Deep Learning Model Comparison

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)

---

## 🌟 Overview

Welcome to the **Electric Production Forecasting** project! This end-to-end Jupyter Notebook (`DL_final.ipynb`) predicts monthly U.S. electric production using historical data from 1985 to 2018. It implements, trains, and evaluates **four different deep learning architectures** to compare their performance on a real-world time-series forecasting problem.

---

## 📊 Dataset Details

| Property | Detail |
|----------|--------|
| 📥 **Source** | [Predict Electricity Consumption — Kaggle](https://www.kaggle.com/code/nageshsingh/predict-electricity-consumption) |
| 📄 **File** | `Electric_Production.csv` |
| 🎯 **Target Variable** | `Value` (Electric Production Index) |
| 📅 **Frequency** | Monthly (Jan 1985 – Jan 2018) |
| 📆 **Date Format** | `MM-DD-YYYY` (e.g. `01-01-1985` = January 1, 1985) |
| 📏 **Total Records** | 397 months |

### ⚙️ Feature Engineering

The following features are engineered to help models capture seasonal and trend patterns in monthly data:

| Feature | Description |
|---------|-------------|
| `lag1` | Previous month's production value |
| `lag12` | Same month last year — captures yearly seasonality |
| `rolling_mean_3` | 3-month rolling average — captures quarterly trend |
| `month_sin` | Sine encoding of month — cyclical seasonality |
| `month_cos` | Cosine encoding of month — cyclical seasonality |
| `quarter` | Quarter of the year (1–4) |

> 🔑 **Why these features?** Monthly data has strong **yearly seasonality** (lag12 captures this) and **quarterly trends** (rolling_mean_3). Day-level features like day_of_week are intentionally excluded as they carry no information in monthly data.

### 📐 Sequence Design

- **Window Size:** 12 months (1 full year of lookback)
- **Train/Test Split:** 80% / 20% (chronological — no leakage)
- **Scalers:** StandardScaler for features, MinMaxScaler for target

---

## 🧠 Models Implemented

All four models process sequences of shape `(12 timesteps × 6 features)` and output a single scalar prediction.

### 1. 🔁 SimpleRNN
A basic recurrent neural network — serves as the baseline. Two stacked layers with Dropout.

### 2. 💾 LSTM (Long Short-Term Memory)
Handles long-range dependencies with memory gates. Two stacked bidirectional layers with recurrent dropout.

### 3. 🚪 GRU (Gated Recurrent Unit)
Lighter than LSTM with fewer parameters. Two stacked bidirectional layers with recurrent dropout.

### 4. 🤖 Vision Transformer (ViT) — via HuggingFace
The time-series sequence is reshaped into a **32×32 pseudo-image** (zero-padded) and fed into a `ViTModel` from HuggingFace Transformers. The **CLS token** output is passed through a regression head for prediction.

```
Sequence (12×6) ──► Flatten ──► Zero-pad ──► Reshape (3×32×32) ──► ViT ──► CLS Token ──► Regressor ──► Prediction
```

> 💡 ViT is an unconventional choice for time-series but demonstrates how attention-based global pattern recognition can be applied to sequential data via image-like representations.

---

## 🏋️ Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Batch Size | 16 (small dataset — more stable) |
| Max Epochs | 150 (RNN/LSTM/GRU) / 50 (ViT) |
| Loss Function | MSE |
| Early Stopping | patience=10, restore best weights |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |

---

## 📈 Results

Models are evaluated on the held-out test set using three metrics:

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — lower is better |
| **MAE** | Mean Absolute Error — lower is better |
| **R²** | Coefficient of Determination — higher is better (max = 1.0) |

> 📌 Exact numbers will vary slightly per run due to random weight initialization.

---

## 📊 Visualizations

The notebook produces three plots:

1. **Bar Chart Comparison** — RMSE, MAE, and R² side-by-side for all 4 models
2. **Predictions vs Actual** — Best model's forecasts overlaid on true values
3. **Convergence Curves** — Validation loss over epochs for all 4 models (log scale)

---

## 🛠️ Dependencies

```bash
pip install tensorflow torch transformers scikit-learn pandas numpy matplotlib
```

| Library | Purpose |
|---------|---------|
| `tensorflow` / `keras` | LSTM, GRU, SimpleRNN models |
| `torch` | Vision Transformer (ViT) training loop |
| `transformers` | HuggingFace ViTConfig + ViTModel |
| `scikit-learn` | Scalers + evaluation metrics |
| `pandas` / `numpy` | Data handling |
| `matplotlib` | Visualization |

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Upload `DL_final.ipynb` and `Electric_Production.csv` to Colab
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Run All cells — everything is self-contained

### Option 2: Local Jupyter
```bash
# Install dependencies
pip install tensorflow torch transformers scikit-learn pandas numpy matplotlib jupyter

# Launch notebook
jupyter notebook DL_final.ipynb
```

> ⚠️ Make sure `Electric_Production.csv` is in the **same directory** as the notebook.
> 📆 Dataset date format is `MM-DD-YYYY` — parsed explicitly as `format='%m-%d-%Y'` in the notebook to avoid ambiguity.

---

## 📁 Repository Structure

```
├── DL_final.ipynb             # Main notebook — all 4 models
├── Electric_Production.csv    # Dataset (monthly, 1985–2018)
└── README.md                  # This file
```

---

## 👨‍💻 Author

| Field | Detail |
|-------|--------|
| **Name** | *(Your Name)* |
| **Roll Number** | *(Your Roll Number)* |
| **Email** | *(Your Email)* |
| **University** | Thapar Institute of Engineering and Technology |

---

## 📝 Notes

- The ViT model uses a **pseudo-image representation** of the time-series — this is an experimental approach and may underperform recurrent models on small monthly datasets
- All recurrent models (LSTM, GRU, RNN) use **2 stacked layers** for depth
- `lag12` is the most important feature for monthly electricity data due to strong **yearly seasonality**
- Results may vary slightly on each run — use the saved comparison table and plots for reproducible reporting
