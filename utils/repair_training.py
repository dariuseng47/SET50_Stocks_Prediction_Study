import os
import pandas as pd
import subprocess
import sys

# Configuration
# Root directory is parent of utils/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKER_LIST_FILE = os.path.join(ROOT_DIR, 'step1_data_import', 'ticker_list.csv')
RESULT_DIR = os.path.join(ROOT_DIR, 'step4_result')
TRAIN_SCRIPT = os.path.join(ROOT_DIR, 'step3_training', 's3_2_run_training.py')

def get_missing_tickers(model_name, all_tickers):
    missing = []
    model_res_dir = os.path.join(RESULT_DIR, model_name)
    if not os.path.exists(model_res_dir):
        return all_tickers
    
    for ticker in all_tickers:
        clean_name = ticker.replace('.', '_')
        pred_file = os.path.join(model_res_dir, f"{clean_name}_predictions.csv")
        if not os.path.exists(pred_file):
            missing.append(ticker)
    return missing

def run_repair():
    # 1. Load all tickers
    tickers_df = pd.read_csv(TICKER_LIST_FILE)
    all_tickers = tickers_df['Ticker'].tolist()
    
    models_to_fix = [
        ('Model1_LSTM', '1'),
        ('Model2_CNN_LSTM', '2')
    ]
    
    for model_name, model_idx in models_to_fix:
        print(f"\n--- Checking {model_name} ---")
        missing = get_missing_tickers(model_name, all_tickers)
        print(f"Missing {len(missing)} tickers: {missing}")
        
        for ticker in missing:
            print(f"🚀 Training {model_name} for {ticker}...")
            try:
                # รันแยกทีละตัวเพื่อให้ชัวร์
                cmd = [sys.executable, TRAIN_SCRIPT, model_idx, 'default', ticker]
                # ใช้ stdout=None เพื่อให้แสดงผลลงหน้าจอตรงๆ
                subprocess.run(cmd, check=True)
                print(f"✅ Finished {ticker}")
            except Exception as e:
                print(f"❌ Failed {ticker}: {e}")

    # 2. Run Evaluation and Visualization at the end
    print("\n--- 📊 Finalizing Results ---")
    scripts = [
        os.path.join(ROOT_DIR, 'step5_evaluation', 's5_2_run_evaluation.py'),
        os.path.join(ROOT_DIR, 'step6_visualization', 's6_1_plot_predictions.py'),
        os.path.join(ROOT_DIR, 'step6_visualization', 's6_2_plot_tables.py'),
        os.path.join(ROOT_DIR, 'step6_visualization', 's6_3_plot_summary.py'),
        os.path.join(ROOT_DIR, 'step6_visualization', 's6_4_error_analysis.py'),
        os.path.join(ROOT_DIR, 'step6_visualization', 's6_5_plot_training_curves.py')
    ]
    
    for s in scripts:
        print(f"Running {os.path.basename(s)}...")
        subprocess.run([sys.executable, s])

if __name__ == "__main__":
    run_repair()
