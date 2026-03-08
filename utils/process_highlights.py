import pandas as pd
import numpy as np
import os

# 1. Load data
# Get root directory relative to this script (utils/ folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(BASE_DIR, 'step4_result', 'summaries', 'Evaluation_Summary_Table.csv')
df = pd.read_csv(input_path)

def clean_val(val):
    if isinstance(val, str):
        return float(val.replace('%', ''))
    return val

# 2. Config
metrics_config = {
    'RMSE': 'min',
    'MAE': 'min',
    'MAPE(%)': 'min',
    'DA(%)': 'max',
    'F1-Score(%)': 'max'
}

# 3. Process
tickers = df['Ticker'].unique()
markdown_content = '# Summary of Evaluation per Ticker (Highlighted Winners)\n\n'
markdown_content += '*Note: **Bold** values represent the best performance for that specific ticker.*\n\n'

highlighted_rows = []

for ticker in tickers:
    temp_df = df[df['Ticker'] == ticker].copy()
    markdown_content += f'### Ticker: {ticker}\n\n'
    
    # Find best values
    bests = {}
    for m, mode in metrics_config.items():
        vals = temp_df[m].apply(clean_val)
        bests[m] = vals.min() if mode == 'min' else vals.max()

    # Table Header
    markdown_content += '| Model | RMSE | MAE | MAPE | DA% | F1-Score |\n'
    markdown_content += '| :--- | :---: | :---: | :---: | :---: | :---: |\n'
    
    for _, row in temp_df.iterrows():
        md_row_parts = [f"{row['Model']}"]
        csv_row = row.to_dict()
        
        for m in ['RMSE', 'MAE', 'MAPE(%)', 'DA(%)', 'F1-Score(%)']:
            current_val = clean_val(row[m])
            is_best = abs(current_val - bests[m]) < 1e-9
            
            if is_best:
                md_row_parts.append(f"**{row[m]}**")
                csv_row[m] = f"{row[m]} *" 
            else:
                md_row_parts.append(f"{row[m]}")
        
        markdown_content += '| ' + ' | '.join(md_row_parts) + ' |\n'
        highlighted_rows.append(csv_row)
    
    markdown_content += '\n---\n'

# 4. Save
output_md_path = os.path.join(BASE_DIR, 'step4_result', 'summaries', 'Highlighted_Results.md')
output_csv_path = os.path.join(BASE_DIR, 'step4_result', 'summaries', 'Evaluation_Summary_Highlighted.csv')

with open(output_md_path, 'w', encoding='utf-8') as f:
    f.write(markdown_content)

pd.DataFrame(highlighted_rows).to_csv(output_csv_path, index=False)
print(f'Done! Saved to {output_md_path} and {output_csv_path}')
