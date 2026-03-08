import pandas as pd
import numpy as np
import os
import glob

# --- Configuration ---
RAW_DATA_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\1_data_import\raw_data'

def calculate_rsi(series, period=14):
    """Calculates RSI for a given series using EMA-based relative strength."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_technical_indicators():
    file_list = glob.glob(os.path.join(RAW_DATA_DIR, "*_stock_data.csv"))
    
    print(f"--- Adding Enhanced Technical Indicators to {len(file_list)} tickers ---")

    for file_path in sorted(file_list):
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            ticker = os.path.basename(file_path)

            # SMA & EMA
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            
            # RSI & Volatility
            df['RSI_14'] = calculate_rsi(df['Close'], period=14)
            if 'Log Return' in df.columns:
                df['Volatility_5'] = df['Log Return'].rolling(window=5).std()
            
            # Save the file
            df.to_csv(file_path)
            print(f"✅ Added Indicators (SMA, EMA, RSI, Vol) for: {ticker}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("--- Feature Engineering Completed ---")

if __name__ == "__main__":
    add_technical_indicators()
