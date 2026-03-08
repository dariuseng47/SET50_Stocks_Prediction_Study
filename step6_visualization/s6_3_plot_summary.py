import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
RESULT_FILE = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step4_result\final_model_comparison.csv'
OUTPUT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step6_visualization\3_global_summary'

def generate_global_summary():
    print(f"--- Generating Model Dominance Summary ---")
    
    if not os.path.exists(RESULT_FILE):
        print(f"❌ Error: {RESULT_FILE} not found.")
        return

    try:
        df = pd.read_csv(RESULT_FILE)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Normalize Model names for consistent display and mapping
    df['Model_Display'] = df['Model']
    
    # Define standard order and colors
    model_order = ['Model1_LSTM', 'Model2_CNN_LSTM', 'Model3_CWGAN_GP', 'Model4_LSTM_CWGAN_GP', 'Model5_LSTM_CNN_CWGAN_GP', 'Model6_CNN_LSTM_CWGAN_GP']
    custom_palette = [
        '#C71585', # Model1_LSTM
        '#800080', # Model2_CNN_LSTM
        '#008080', # Model3_CWGAN_GP
        '#20B2AA', # Model4_LSTM_CWGAN_GP
        '#3CB371', # Model5_LSTM_CNN_CWGAN_GP
        '#32CD32'  # Model6_CNN_LSTM_CWGAN_GP
    ]
    color_map = dict(zip(model_order, custom_palette))

    # Metrics configuration: {Metric_Name_in_CSV: Higher_is_better_bool}
    # Note: CSV columns are 'RMSE', 'MAE', 'MAPE(%)', 'DA(%)', 'F1-Score'
    metrics_mapping = {
        'RMSE': False,
        'MAE': False,
        'MAPE(%)': False,
        'DA(%)': True,
        'F1-Score': True
    }

    all_wins = []
    num_stocks = df['Ticker'].nunique()

    for metric, higher_is_better in metrics_mapping.items():
        if metric not in df.columns:
            continue
            
        # Group by Ticker and find the winner for this metric
        if higher_is_better:
            # For DA and F1, max is better
            winners = df.loc[df.groupby('Ticker')[metric].idxmax()]
        else:
            # For RMSE, MAE, MAPE, min is better
            winners = df.loc[df.groupby('Ticker')[metric].idxmin()]
        
        # Count wins per model display name
        counts = winners['Model_Display'].value_counts().reindex(model_order, fill_value=0)
        
        for model, count in counts.items():
            display_metric = metric.replace('(%)', '').replace('-Score', '')
            all_wins.append({
                'Metric': display_metric,
                'Model': model,
                'Win_Count': count
            })

    win_df = pd.DataFrame(all_wins)

    # Plotting
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 8))
    
    # We use win_df and explicitly set hue_order to ensure color mapping is correct
    ax = sns.barplot(
        data=win_df, 
        x='Metric', 
        y='Win_Count', 
        hue='Model', 
        hue_order=model_order,
        palette=color_map
    )

    plt.title(f'Model Dominance: Number of "Best Result" Wins per Metric (Across {num_stocks} Stocks)', fontsize=18, fontweight='bold', pad=25)
    plt.ylabel('Number of Stocks (Wins)', fontsize=14)
    plt.xlabel('Evaluation Metrics', fontsize=14)
    
    # Dynamic Y-axis based on number of stocks
    y_limit = max(win_df['Win_Count'].max() + 2, num_stocks + 1)
    plt.yticks(range(0, int(y_limit) + 1, max(1, int(y_limit // 10))))
    plt.ylim(0, y_limit)
    
    plt.legend(title='Model Architecture', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Add labels on bars
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=10, fontweight='bold')

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'Leaderboard_Overall.png')
    plt.savefig(output_path, dpi=200)
    
    # Also save the original filename requested if it was Leaderboard_RMSE.png
    # But usually a combined plot is better as requested.
    plt.savefig(os.path.join(OUTPUT_DIR, 'Leaderboard_RMSE.png'), dpi=200)
    
    plt.close()
    print(f"✅ Saved dominance plot to: {output_path}")

if __name__ == "__main__":
    generate_global_summary()
