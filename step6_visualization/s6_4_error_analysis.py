import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

# --- Configuration ---
RESULT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step4_result'
OUTPUT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step6_visualization\5_error_analysis'

def run_error_analysis():
    model_dirs = [d for d in os.listdir(RESULT_DIR) if os.path.isdir(os.path.join(RESULT_DIR, d))]
    
    for model_name in model_dirs:
        print(f"--- Analyzing Errors for Model: {model_name} ---")
        save_path = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(save_path, exist_ok=True)
        
        files = glob.glob(os.path.join(RESULT_DIR, model_name, "*_predictions.csv"))
        
        for file in files:
            ticker = os.path.basename(file).replace('_predictions.csv', '')
            df = pd.read_csv(file)
            actual = df['Actual']
            predicted = df['Predicted']
            error = actual - predicted

            # 1. Scatter Plot (Actual vs Predicted)
            plt.figure(figsize=(8, 8))
            plt.scatter(actual, predicted, alpha=0.5, color='green')
            # วาดเส้นทแยงมุม 45 องศา (เส้นสมบูรณ์แบบ)
            max_val = max(actual.max(), predicted.max())
            min_val = min(actual.min(), predicted.min())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            plt.title(f"Actual vs Predicted: {ticker} ({model_name})")
            plt.xlabel("Actual Price")
            plt.ylabel("Predicted Price")
            plt.grid(True)
            plt.savefig(os.path.join(save_path, f"{ticker}_scatter.png"))
            plt.close()

            # 2. Error Distribution (Histogram)
            plt.figure(figsize=(10, 6))
            sns.histplot(error, kde=True, color='purple')
            plt.axvline(x=0, color='black', linestyle='--')
            plt.title(f"Error Distribution: {ticker} ({model_name})")
            plt.xlabel("Price Error (Actual - Predicted)")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(save_path, f"{ticker}_error_hist.png"))
            plt.close()

            print(f"✅ Analysis saved for {ticker}")

if __name__ == "__main__":
    run_error_analysis()
