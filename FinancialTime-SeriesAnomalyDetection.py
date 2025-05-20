# financial_anomaly_detection.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from prophet import Prophet

# ----------------------------------------
# Step 1: Download Historical Stock Data
# ----------------------------------------
def fetch_stock_data(ticker='AAPL', start='2020-01-01', end='2024-12-31'):
    data = yf.download(ticker, start=start, end=end)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.dropna(inplace=True)
    return data

# ----------------------------------------
# Step 2: Add Financial Indicators
# ----------------------------------------
def add_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['STD_20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['STD_20'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['STD_20'] * 2)
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df.dropna(inplace=True)
    return df

# ----------------------------------------
# Step 3: Anomaly Detection (Isolation Forest)
# ----------------------------------------
def detect_anomalies(df):
    features = df[['Close', 'Volume', 'SMA_20', 'EMA_20', 'RSI']].copy()
    features.fillna(method='bfill', inplace=True)
    model = IsolationForest(contamination=0.01, random_state=42)
    df['Anomaly'] = model.fit_predict(features)
    df['Anomaly'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)
    return df

# ----------------------------------------
# Step 4: Time Series Forecasting with Prophet
# ----------------------------------------
def forecast_with_prophet(df):
    df_prophet = df.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    return model, forecast

# ----------------------------------------
# Step 5: Visualization
# ----------------------------------------
def plot_anomalies(df, ticker='AAPL'):
    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.6)
    plt.scatter(df[df['Anomaly'] == 1].index, df[df['Anomaly'] == 1]['Close'],
                color='red', label='Anomaly', s=50)
    plt.title(f'{ticker} - Price with Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_forecast(model, forecast):
    model.plot(forecast)
    plt.title("Prophet Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------------------
# Main Execution
# ----------------------------------------
if __name__ == "__main__":
    ticker = 'AAPL'
    stock_data = fetch_stock_data(ticker)
    stock_data = add_indicators(stock_data)
    stock_data = detect_anomalies(stock_data)

    print(stock_data[['Close', 'Anomaly']].tail())

    plot_anomalies(stock_data, ticker)

    prophet_model, forecast = forecast_with_prophet(stock_data)
    plot_forecast(prophet_model, forecast)
