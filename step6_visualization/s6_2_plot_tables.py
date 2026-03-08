import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# --- Configuration ---
RESULT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step4_result'
OUTPUT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step6_visualization\2_stock_tables'

MODEL_ORDER = ['Model1_LSTM', 'Model2_CNN_LSTM', 'Model3_CWGAN_GP', 'Model4_LSTM_CWGAN_GP', 'Model5_LSTM_CNN_CWGAN_GP', 'Model6_CNN_LSTM_CWGAN_GP']

def render_mpl_table(data, col_width=3.5, row_height=0.8, font_size=12,
                     header_color='#008080', row_colors=['#f8f9fa', 'w'], edge_color='#dddddd',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0:
            cell.set_text_props(weight='bold', color='w', fontsize=font_size+2)
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
            cell.set_text_props(fontsize=font_size)
    return ax

def generate_stock_tables():
    all_results = []
    result_files = glob.glob(os.path.join(RESULT_DIR, "**/evaluation_results.txt"), recursive=True)
    
    for file in result_files:
        model_name = os.path.basename(os.path.dirname(file))
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split(', ')
                res_dict = {'Model': model_name}
                for p in parts:
                    key, val = p.split(': ')
                    res_dict[key.strip()] = val.strip()
                all_results.append(res_dict)
    
    if not all_results: return

    df_full = pd.DataFrame(all_results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tickers = df_full['Ticker'].unique()
    for ticker in tickers:
        stock_df = df_full[df_full['Ticker'] == ticker].copy()
        
        # จัดเรียงลำดับโมเดลตามที่กำหนด
        stock_df['Order'] = stock_df['Model'].apply(lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99)
        stock_df = stock_df.sort_values('Order').drop(columns=['Order', 'Ticker'])
        
        # จัดรูปแบบตัวเลขและ Column Name
        for col in ['RMSE', 'MSE', 'MAE']:
            if col in stock_df.columns:
                stock_df[col] = stock_df[col].astype(float).map('{:.4f}'.format)
        
        # เรนเดอร์ตารางเป็นรูปภาพ
        ax = render_mpl_table(stock_df)
        plt.title(f"Performance Comparison - {ticker}", fontsize=16, y=1.02)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_table.png"), bbox_inches='tight', dpi=150)
        plt.close()
        print(f"✅ Generated styled table for {ticker}")

if __name__ == "__main__":
    generate_stock_tables()
