# Financial Time-Series Anomaly Detection

This project identifies anomalies in historical stock price data using statistical indicators, unsupervised learning, and time-series forecasting.

## 📌 Objectives

- Detect unusual stock price movements using **Isolation Forest**
- Visualize **anomalies** in stock prices over time
- Forecast future stock prices using **Facebook Prophet**

## 📂 Files

- `financial_anomaly_detection.py` – Main script containing all steps
- No dataset required upfront – uses `yfinance` to download historical data

## 🧪 Requirements

Install dependencies using pip:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn prophet
