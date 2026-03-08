import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# --- Configuration ---
RESULT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step4_result'
OUTPUT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step6_visualization\4_training_curves'

def plot_training_curves():
    model_dirs = [d for d in os.listdir(RESULT_DIR) if os.path.isdir(os.path.join(RESULT_DIR, d))]
    
    for model_name in model_dirs:
        print(f"--- Plotting Training Curves for Model: {model_name} ---")
        save_path = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(save_path, exist_ok=True)
        
        history_files = glob.glob(os.path.join(RESULT_DIR, model_name, "*_training_history.csv"))
        
        for file in history_files:
            ticker = os.path.basename(file).replace('_training_history.csv', '')
            df = pd.read_csv(file)
            
            # 1. Plot Loss
            plt.figure(figsize=(10, 6))
            if 'loss' in df.columns: # Non-GAN
                plt.plot(df['loss'], label='Train Loss')
                if 'val_loss' in df.columns:
                    plt.plot(df['val_loss'], label='Val Loss')
            elif 'd_loss' in df.columns: # GAN
                plt.plot(df['d_loss'], label='D Loss')
                plt.plot(df['g_loss'], label='G Loss')
            
            plt.title(f"Training Loss: {ticker} ({model_name})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(save_path, f"{ticker}_loss_curve.png"))
            plt.close()

            # 2. Plot MAE/Metric if available
            if 'mae' in df.columns or 'mean_absolute_error' in df.columns:
                plt.figure(figsize=(10, 6))
                col = 'mae' if 'mae' in df.columns else 'mean_absolute_error'
                plt.plot(df[col], label='MAE', color='orange')
                plt.title(f"Training MAE: {ticker} ({model_name})")
                plt.xlabel("Epoch")
                plt.ylabel("MAE")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(save_path, f"{ticker}_mae_curve.png"))
                plt.close()
            
            print(f"✅ Curves saved for {ticker}")

if __name__ == "__main__":
    plot_training_curves()
