# 📈 Comparative Study of Deep Learning & GAN Architectures for Thai Stock Prediction (SET50)

A comprehensive research framework for evaluating and benchmarking multiple state-of-the-art Deep Learning models for predicting Thai stock prices (SET50). This project compares 6 different architectures, from traditional Recurrent Neural Networks to advanced Generative Adversarial Networks (GANs).

---

## 📑 Project Overview

This study focuses on predicting the daily closing prices of 18 major Thai stocks (SET50) using historical data from 2019 to 2024. The primary goal is to compare the performance of baseline regression models against generative models (GANs) equipped with stability mechanisms like **WGAN-GP** and accuracy constraints like **L1 Reconstruction Loss**.

### Key Technical Goals:
- Implementation of **Conditional Wasserstein GAN with Gradient Penalty (CWGAN-GP)**.
- Utilization of **L1 Loss (MAE)** to balance visual realism and numerical accuracy.
- Comparison of hybrid architectures (CNN-LSTM) within a GAN framework.
- Robust validation using **Walk-forward Validation** and **Static Splits**.

---

## 📊 Dataset & Feature Engineering

### 1. Selection Criteria (1,212 Trading Days)
To ensure statistical significance, 18 stocks from the SET50 index were selected based on:
- **Continuity:** No IPOs, splits, or major ticker changes during the study period.
- **Integrity:** Exactly 1,212 trading days (July 2019 – July 2024).
- **Selected Tickers:** `ADVANC`, `AOT`, `BBL`, `BDMS`, `CPALL`, `CPN`, `GPSC`, `HMPRO`, `IVL`, `KBANK`, `KTB`, `MINT`, `PTTEP`, `PTT`, `SCC`, `TRUE`, `TTB`, `WHA`.

### 2. Feature Set
- **Input Window:** 5-day lookback period (t-4 to t).
- **Target:** Closing Price at t+1.
- **Technical Indicators:**
    - **Trend:** SMA (5, 10), EMA (5, 10)
    - **Momentum:** RSI (14)
    - **Volatility:** 5-day Rolling Standard Deviation
    - **Returns:** Daily Return, Return %, and Log Return

---

## 🏗️ Model Architectures

| Model | Type | Architecture Highlights |
| :--- | :--- | :--- |
| **Model 1** | **LSTM** | 2 LSTM Layers (50 units), 1 Dense (25), Dropout 0.2 |
| **Model 2** | **CNN-LSTM** | 1 Conv1D (64 filters), 2 LSTM (50), 1 Dense (25) |
| **Model 3** | **CWGAN-GP** | Dense Generator (128, 256), Dense Discriminator (128, 64) |
| **Model 4** | **LSTM-CWGAN-GP** | 3-Layer LSTM Generator, 2-Layer LSTM Discriminator |
| **Model 5** | **LSTM-CNN-GAN** | LSTM Generator (3 layers), CNN Discriminator (64 filters) |
| **Model 6** | **CNN-LSTM-GAN** | Hybrid CNN-LSTM Generator, LSTM Discriminator |

### Technical Innovations:
- **Stability:** Gradient Penalty (λ_gp = 10.0) and Critic Ratio (n_critic = 5).
- **Accuracy:** L1 Reconstruction Weight (λ_L1 = 100.0).
- **Robustness:** `LeakyReLU` (negative_slope=0.1) and `tanh` activation for scaled outputs.

---

## 🚀 Execution Pipeline

Follow this order for consistent results:

### Step 1: Data Preparation (`step1_data_import/`)
1. `s1_1_yfinance_loader.py`: Download raw data and calculate returns.
2. `s1_2_check_data_quality.py`: Verify 1,212-day requirement.
3. `s1_5_feature_engineering.py`: Add technical indicators.

### Step 2: Training (`step3_training/`)
- Run `s3_2_run_training.py`: **Interactive CLI.** Select your model (1-6) and feature set.
- GAN models train for 1,000 epochs; Baselines for 100 epochs with Early Stopping.

### Step 3: Evaluation & Visualization
- `s5_2_run_evaluation.py`: Calculate RMSE, MAE, MAPE, and Directional Accuracy.
- `step6_visualization/`: Scripts to generate Price Plots, Leaderboards, and Training Curves.

---

## 📂 Project Structure

```text
├── docs/                      # Technical methodology, logs, and full reports
├── utils/                     # Maintenance & path-fixing scripts
├── step1_data_import/         # Data fetching and technical analysis
├── step2_model_architecture/  # Keras model definitions
├── step3_training/            # Training controllers and WGAN-GP logic
├── step4_result/              # Raw CSV outputs
│   └── summaries/             # Performance leaderboards and markdown reports
├── step5_evaluation/          # Metric calculation (Regression & Directional)
├── step6_visualization/       # 7 Categories of automated PNG charts
├── config.py                  # Global hyperparameters
├── run.py                     # Main entry point
└── requirements.txt           # Environment dependencies
```

---

## 📊 Evaluation Metrics
The models are evaluated on 8 performance metrics:
1. **RMSE** (Root Mean Square Error)
2. **MSE** (Mean Square Error)
3. **MAE** (Mean Absolute Error)
4. **MAPE** (Mean Absolute Percentage Error)
5. **DA** (Directional Accuracy %)
6. **Precision**, **Recall**, **F1-Score**

---

## 📄 License & Credits
- **License:** MIT License
- **Author:** Khunphon Somjitr
- **Frameworks:** TensorFlow/Keras, Pandas, NumPy, yfinance, Matplotlib.
