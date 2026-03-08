import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
TICKER_LIST_FILE = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\1_data_import\ticker_list.csv'
START_DATE = '2019-07-19'
END_DATE = '2024-07-23' # To include July 22
OUTPUT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\1_data_import\raw_data'

def download_and_calculate_returns():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(TICKER_LIST_FILE):
        print(f"Error: {TICKER_LIST_FILE} not found!")
        return
        
    tickers_df = pd.read_csv(TICKER_LIST_FILE)
    tickers = tickers_df['Ticker'].tolist()
    
    print(f"--- Processing {len(tickers)} tickers with Enhanced Return Calculations ---")

    for ticker in tickers:
        try:
            print(f"Downloading & Calculating for {ticker}...")
            df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=False)
            
            if df.empty:
                print(f"Warning: No data found for {ticker}")
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Ensure Adj Close exists
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            
            # --- Enhanced Calculations ---
            # 1. Daily Return = Current Adj Close - Previous Adj Close
            df['Daily Return'] = df['Adj Close'].diff()
            
            # 2. Daily Return % = (Current Adj Close / Previous Adj Close) - 1 * 100
            df['Daily Return %'] = df['Adj Close'].pct_change() * 100
            
            # 3. Log Return = ln(Current Adj Close / Previous Adj Close)
            # We use np.log(df['Adj Close'] / df['Adj Close'].shift(1))
            df['Log Return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
            
            # Select and reorder columns
            cols_to_keep = [
                'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                'Daily Return', 'Daily Return %', 'Log Return'
            ]
            df = df[[c for c in cols_to_keep if c in df.columns]]
            
            # Drop the first row which will have NaN for all return calculations
            df = df.dropna()
            
            clean_name = ticker.replace('.', '_')
            file_path = os.path.join(OUTPUT_DIR, f"{clean_name}_stock_data.csv")
            
            df.to_csv(file_path)
            print(f"Successfully saved with log returns to {file_path}")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    print("--- Download and Enhanced Return Calculation Completed ---")

if __name__ == "__main__":
    download_and_calculate_returns()
