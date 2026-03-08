import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# --- Configuration ---
RESULT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step4_result'
OUTPUT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step6_visualization\1_prediction_plots'

# สีประจำโมเดลตาม First Study
MODEL_COLORS = {
    'Model1_LSTM': '#C71585',
    'Model2_CNN_LSTM': '#800080',
    'Model3_CWGAN_GP': '#008080',
    'Model4_LSTM_CWGAN_GP': '#20B2AA',
    'Model5_LSTM_CNN_CWGAN_GP': '#3CB371',
    'Model6_CNN_LSTM_CWGAN_GP': '#32CD32'
}

def plot_all_predictions():
    if not os.path.exists(RESULT_DIR):
        print(f"❌ ไม่พบโฟลเดอร์ {RESULT_DIR}")
        return

    model_dirs = [d for d in os.listdir(RESULT_DIR) if os.path.isdir(os.path.join(RESULT_DIR, d))]
    
    for model_name in model_dirs:
        print(f"--- Plotting for Model: {model_name} ---")
        save_path = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(save_path, exist_ok=True)
        
        # ค้นหาสี (ถ้าไม่เจอใช้สีเทา)
        line_color = MODEL_COLORS.get(model_name, '#808080')
        
        files = glob.glob(os.path.join(RESULT_DIR, model_name, "*_predictions.csv"))
        
        for file in files:
            ticker = os.path.basename(file).replace('_predictions.csv', '')
            df = pd.read_csv(file)
            
            plt.figure(figsize=(14, 7))
            plt.plot(df['Actual'], label='Actual Price', color='black', linewidth=2, alpha=0.7)
            plt.plot(df['Predicted'], label=f'Predicted ({model_name})', color=line_color, linestyle='--', linewidth=1.5)
            
            plt.title(f"Stock Price Prediction: {ticker} ({model_name})", fontsize=14)
            plt.xlabel("Days", fontsize=12)
            plt.ylabel("Price (THB)", fontsize=12)
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            
            plt.savefig(os.path.join(save_path, f"{ticker}_plot.png"))
            plt.close()
            print(f"✅ Saved plot for {ticker}")

if __name__ == "__main__":
    plot_all_predictions()
