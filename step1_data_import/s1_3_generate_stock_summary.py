import pandas as pd
import numpy as np
import os
import glob

# --- Configuration ---
RAW_DATA_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step1_data_import\raw_data'
OUTPUT_FILE = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step1_data_import\stock_summary.csv'

# ข้อมูลชื่อบริษัท (19 ตัวที่เหลือ)
COMPANY_NAMES = {
    'ADVANC_BK': 'Advanced Info Service PCL',
    'AOT_BK': 'Airports of Thailand PCL',
    'BBL_BK': 'Bangkok Bank PCL',
    'BDMS_BK': 'Bangkok Dusit Medical Services PCL',
    'CPALL_BK': 'CP ALL PCL',
    'CPN_BK': 'Central Pattana PCL',
    'GPSC_BK': 'Global Power Synergy PCL',
    'HMPRO_BK': 'Home Product Center PCL',
    'IVL_BK': 'Indorama Ventures PCL',
    'KBANK_BK': 'Kasikornbank PCL',
    'KTB_BK': 'Krung Thai Bank PCL',
    'MINT_BK': 'Minor International PCL',
    'PTTEP_BK': 'PTT Exploration and Production PCL',
    'PTT_BK': 'PTT PCL',
    'SCB_BK': 'SCB X PCL',
    'SCC_BK': 'Siam Cement PCL',
    'TRUE_BK': 'True Corporation PCL',
    'TTB_BK': 'TMBThanachart Bank PCL',
    'WHA_BK': 'WHA Corporation PCL'
}

def generate_summary():
    file_list = glob.glob(os.path.join(RAW_DATA_DIR, "*_stock_data.csv"))
    summary_data = []

    print(f"--- Generating Summary for {len(file_list)} tickers ---")

    # Sort file_list to maintain consistent order
    for i, file_path in enumerate(sorted(file_list), 1):
        try:
            df = pd.read_csv(file_path)
            ticker_id = os.path.basename(file_path).replace('_stock_data.csv', '')
            
            # 1. Average Price (5Y)
            avg_price = df['Close'].mean()
            
            # 2. Annual Volatility (Std Dev of Log Return * sqrt(252))
            # Log Return index in df depends on how it was saved, check if exists
            if 'Log Return' in df.columns:
                daily_std = df['Log Return'].std()
                annual_vol = daily_std * np.sqrt(252)
            else:
                annual_vol = 0
                
            # 3. Average Volume
            avg_vol = df['Volume'].mean()
            
            # 4. Get Company Name
            company_name = COMPANY_NAMES.get(ticker_id, "N/A")

            summary_data.append({
                'no.': i,
                'ticker': ticker_id.replace('_', '.'),
                'company name': company_name,
                'average price(5Y)': round(avg_price, 2),
                'annual volatility': round(annual_vol, 4),
                'average volume': int(avg_vol)
            })
            print(f"Processed: {ticker_id}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Create DataFrame and Save
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_FILE, index=False)
    print(f"--- Summary Report Successfully Saved to {OUTPUT_FILE} ---")

if __name__ == "__main__":
    generate_summary()
