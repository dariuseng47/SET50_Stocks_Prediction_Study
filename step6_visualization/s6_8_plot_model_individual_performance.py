import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
RESULT_FILE = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step4_result\final_model_comparison.csv'
OUTPUT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step6_visualization\6_individual_model_performance'

def plot_individual_model_performance():
    print(f"--- Generating Individual Performance Plots for each Model ---")
    
    if not os.path.exists(RESULT_FILE):
        print(f"❌ Error: {RESULT_FILE} not found.")
        return

    df = pd.read_csv(RESULT_FILE)
    
    # 1. หาค่า Global Max เพื่อตั้งค่าแกน Y ให้เท่ากันทุกรูป
    # หาค่าสูงสุดของ RMSE หรือ MAE
    global_max_error = max(df['RMSE'].max(), df['MAE'].max()) * 1.1 # เผื่อพื้นที่ด้านบน 10%
    # หาค่าสูงสุดของ MAPE
    global_max_mape = df['MAPE(%)'].max() * 1.1
    
    # Define standard order
    model_order = ['Model1_LSTM', 'Model2_CNN_LSTM', 'Model3_CWGAN_GP', 'Model4_LSTM_CWGAN_GP', 'Model5_LSTM_CNN_CWGAN_GP', 'Model6_CNN_LSTM_CWGAN_GP']
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name in model_order:
        model_df = df[df['Model'] == model_name].sort_values('Ticker')
        if model_df.empty:
            continue
            
        print(f"Processing {model_name}...")
        
        # ปรับรูปแบบชื่อหัวข้อ: Model1_LSTM -> Model 1 : LSTM
        display_name = model_name.replace('Model', 'Model ').replace('_', ' : ', 1).replace('_', '-')
        
        fig, ax1 = plt.subplots(figsize=(16, 8))
        
        # 1. Plot RMSE and MAE as grouped bars
        x = range(len(model_df))
        width = 0.35
        
        bar1 = ax1.bar([i - width/2 for i in x], model_df['RMSE'], width, label='RMSE (THB)', color='#4682B4', alpha=0.8)
        bar2 = ax1.bar([i + width/2 for i in x], model_df['MAE'], width, label='MAE (THB)', color='#FFA07A', alpha=0.8)
        
        ax1.set_xlabel('Stock Tickers', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Error Value (THB)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Performance Profile: {display_name}', fontsize=20, fontweight='bold', pad=20)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_df['Ticker'], rotation=45, ha='right')
        ax1.set_ylim(0, global_max_error) # Fix Y1 Axis
        ax1.legend(loc='upper left')
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        # 2. Plot MAPE on secondary axis
        ax2 = ax1.twinx()
        line1 = ax2.plot(x, model_df['MAPE(%)'], color='#D2691E', marker='o', linewidth=2, label='MAPE (%)')
        ax2.set_ylabel('Percentage Error (MAPE %)', fontsize=12, color='#D2691E', fontweight='bold')
        ax2.set_ylim(0, global_max_mape) # Fix Y2 Axis
        ax2.tick_params(axis='y', labelcolor='#D2691E')
        
        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f'Performance_{model_name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
    print(f"✅ All 6 plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    plot_individual_model_performance()
