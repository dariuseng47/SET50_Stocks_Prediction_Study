import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
RESULT_FILE = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step4_result\final_model_comparison.csv'
OUTPUT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step6_visualization\7_individual_model_direction'

def plot_individual_model_direction():
    print(f"--- Generating Individual Directional Plots for each Model ---")
    
    if not os.path.exists(RESULT_FILE):
        print(f"❌ Error: {RESULT_FILE} not found.")
        return

    df = pd.read_csv(RESULT_FILE)
    
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
        
        # คัดกรองและปรับสเกล F1 ให้เป็น % เหมือน DA เพื่อความสวยงามในกราฟ
        plot_df = model_df.copy()
        plot_df['F1_Percent'] = plot_df['F1-Score'] * 100
        
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.set_style("whitegrid")
        
        x = range(len(plot_df))
        width = 0.35
        
        # Plot DA% and F1% and capture containers
        rects1 = ax.bar([i - width/2 for i in x], plot_df['DA(%)'], width, label='Directional Accuracy (DA %)', color='#66CDAA', alpha=0.8)
        rects2 = ax.bar([i + width/2 for i in x], plot_df['F1_Percent'], width, label='F1-Score (%)', color='#9370DB', alpha=0.8)
        
        # Add labels on top of bars
        ax.bar_label(rects1, padding=3, fmt='%.2f%%', fontsize=8, fontweight='bold', rotation=90)
        ax.bar_label(rects2, padding=3, fmt='%.2f%%', fontsize=8, fontweight='bold', rotation=90)
        
        ax.set_xlabel('Stock Tickers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Direction Capability: {display_name}', fontsize=20, fontweight='bold', pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['Ticker'], rotation=45, ha='right')
        ax.set_yticks(range(0, 101, 10))
        ax.set_ylim(0, 100) # แกน Y คงที่ 0-100%
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        # เพิ่มเส้นค่าเฉลี่ยอ้างอิง 50% (Random Guess Baseline)
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Random Guess (50%)')

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f'Direction_{model_name}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        
    print(f"✅ All 6 directional plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    plot_individual_model_direction()
