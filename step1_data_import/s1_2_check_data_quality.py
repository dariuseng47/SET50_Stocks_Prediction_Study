import pandas as pd
import numpy as np
import os
import glob

# --- Configuration ---
RAW_DATA_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\1_data_import\raw_data'
OUTPUT_REPORT = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\1_data_import\data_quality_report.txt'

def check_quality():
    file_list = glob.glob(os.path.join(RAW_DATA_DIR, "*_stock_data.csv"))
    report_lines = []
    
    report_lines.append("--- 🔍 Data Quality Report (3rd Study) ---")
    report_lines.append(f"Check Date: {pd.Timestamp.now()}\n")
    
    stats = []
    for file_path in sorted(file_list):
        try:
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            ticker = os.path.basename(file_path).replace('_stock_data.csv', '')
            nan_count = df.isnull().sum().sum()
            zero_vol_days = (df['Volume'] == 0).sum()
            stats.append({
                'ticker': ticker,
                'rows': len(df),
                'nan_count': nan_count,
                'zero_vol_days': zero_vol_days,
                'start': df.index.min().date(),
                'end': df.index.max().date()
            })
        except Exception as e:
            report_lines.append(f"Error processing {file_path}: {e}")

    stats_df = pd.DataFrame(stats)
    
    # Check Row counts
    counts = stats_df['rows'].unique()
    if len(counts) == 1:
        report_lines.append(f"✅ PASS: All {len(stats_df)} tickers have exactly {counts[0]} rows.")
    else:
        report_lines.append(f"⚠️ WARNING: Row mismatch found! Counts: {counts}")
        mode_rows = stats_df['rows'].mode()[0]
        outliers = stats_df[stats_df['rows'] != mode_rows]
        for _, row in outliers.iterrows():
            report_lines.append(f"   - {row['ticker']}: {row['rows']} rows (Expected {mode_rows})")

    # Check Missing values
    total_nan = stats_df['nan_count'].sum()
    if total_nan == 0:
        report_lines.append("✅ PASS: No NaN values found.")
    else:
        report_lines.append(f"⚠️ WARNING: {total_nan} missing values detected.")

    # Check Zero volume
    total_zero_vol = stats_df['zero_vol_days'].sum()
    if total_zero_vol == 0:
        report_lines.append("✅ PASS: No Zero Volume (Suspended) days found.")
    else:
        report_lines.append(f"ℹ️ INFO: Found {total_zero_vol} days with Zero Volume.")
        
    report_lines.append("\n" + stats_df.to_string(index=False))

    report_content = "\n".join(report_lines)
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(report_content)

if __name__ == "__main__":
    check_quality()
