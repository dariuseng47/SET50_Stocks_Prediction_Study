import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_stock_data(file_path, lookback=5, train_split_ratio=0.8, feature_cols=['Close']):
    """
    Standard Preprocessor (Static Split) - Used mainly for GANs or simple tests.
    """
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        for col in feature_cols:
            if col not in df.columns: return None, None, None, None, None, None, None
        df = df.dropna()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None, None, None, None, None

    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_scaled = scaler_X.fit_transform(df[feature_cols])
    y_scaled = scaler_y.fit_transform(df[['Close']])

    X, y = [], []
    all_dates = df.index[lookback:].tolist()
    
    for i in range(lookback, len(df)):
        X.append(X_scaled[i-lookback:i, :])
        y.append(y_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    train_len = int(len(X) * train_split_ratio)
    
    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]
    test_dates = all_dates[train_len:]
    y_test_actual = scaler_y.inverse_transform(y[train_len:].reshape(-1, 1))

    return X_train, y_train, X_test, y_test_actual, scaler_X, scaler_y, test_dates

def get_walk_forward_data(file_path, window_size=252, step_size=21, lookback=5, feature_cols=['Close']):
    """
    Generator function for Walk-forward windows.
    Returns: (X_train, y_train, X_test, y_test_actual, scaler_y, test_dates) for each step.
    """
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        df = df.dropna()
    except Exception as e:
        print(f"Error: {e}")
        return

    # ลูปเลื่อนหน้าต่างไปเรื่อยๆ
    # เริ่มจาก window_size แรก ไปจนถึงจุดที่ข้อมูลไม่พอสำหรับ step_size ถัดไป
    for start_idx in range(0, len(df) - window_size - step_size + 1, step_size):
        train_df = df.iloc[start_idx : start_idx + window_size]
        test_df = df.iloc[start_idx + window_size - lookback : start_idx + window_size + step_size]
        
        # Scaling (Fit เฉพาะ Training เท่านั้น เพื่อป้องกัน Data Leakage)
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
        # Fit train
        X_train_raw = scaler_X.fit_transform(train_df[feature_cols])
        y_train_raw = scaler_y.fit_transform(train_df[['Close']])
        
        # Transform test
        X_test_raw = scaler_X.transform(test_df[feature_cols])
        y_test_raw = scaler_y.transform(test_df[['Close']])
        
        # Create Sequences
        def create_seq(data_X, data_y, offset=0):
            xs, ys = [], []
            for i in range(lookback, len(data_X)):
                xs.append(data_X[i-lookback:i, :])
                ys.append(data_y[i, 0])
            return np.array(xs), np.array(ys)

        X_train, y_train = create_seq(X_train_raw, y_train_raw)
        X_test, y_test_scaled = create_seq(X_test_raw, y_test_raw)
        
        # วันที่และราคาจริงสำหรับการวัดผล
        test_dates = test_df.index[lookback:].tolist()
        y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))
        
        yield X_train, y_train, X_test, y_test_actual, scaler_y, test_dates

if __name__ == "__main__":
    print("Testing Preprocessor Module...")
