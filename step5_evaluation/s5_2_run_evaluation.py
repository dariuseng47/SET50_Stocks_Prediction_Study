import os
import sys
import pandas as pd
import numpy as np
import glob

# เพิ่ม Path หลักเพื่อให้หา Module เจอ
root_path = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study'
sys.path.append(root_path)

# Import ฟังก์ชันคำนวณจากไฟล์ s5_1_metrics
from step5_evaluation.s5_1_metrics import calculate_all_metrics

# --- Configuration ---
RESULT_DIR = os.path.join(root_path, 'step4_result')
OUTPUT_FILE = os.path.join(root_path, 'step4_result', 'final_model_comparison.csv')

def run_evaluation():
    if not os.path.exists(RESULT_DIR):
        print(f"❌ ไม่พบโฟลเดอร์ผลลัพธ์ที่: {RESULT_DIR}")
        return

    # 1. รายชื่อโมเดล (โฟลเดอร์ย่อยใน step4_result)
    model_folders = sorted([d for d in os.listdir(RESULT_DIR) if os.path.isdir(os.path.join(RESULT_DIR, d))])
    
    # 2. ค้นหารายชื่อหุ้นทั้งหมดที่มีการพยากรณ์ไว้
    all_tickers = set()
    for model in model_folders:
        files = glob.glob(os.path.join(RESULT_DIR, model, "*_predictions.csv"))
        for f in files:
            ticker = os.path.basename(f).split('_')[0]
            all_tickers.add(ticker)
    
    sorted_tickers = sorted(list(all_tickers))
    print(f"--- 📊 Starting Fair Evaluation for {len(sorted_tickers)} tickers ---")

    final_report_data = []

    for ticker in sorted_tickers:
        print(f"Evaluating {ticker}...")
        
        # ค้นหาช่วงวันที่ร่วมกัน (Common Dates) 
        # เนื่องจาก GAN มีข้อมูลน้อยกว่า (20% ท้าย) เราจะใช้ผลลัพธ์ของโมเดล GAN เป็นตัวตั้งต้น
        gan_files = glob.glob(os.path.join(RESULT_DIR, "*GAN*", f"{ticker}_*predictions.csv"))
        
        if not gan_files:
            print(f"   ⚠️ Warning: No GAN results for {ticker}, skipping fair alignment.")
            continue
            
        # ดึงวันที่จากไฟล์พยากรณ์ของ GAN ตัวแรก
        ref_df = pd.read_csv(gan_files[0])
        common_dates = set(ref_df['Date'].unique())
        
        for model_name in model_folders:
            # ค้นหาไฟล์ของโมเดลนี้สำหรับหุ้นตัวนี้
            pattern = os.path.join(RESULT_DIR, model_name, f"{ticker}_*predictions.csv")
            matches = glob.glob(pattern)
            
            if not matches:
                continue
                
            df = pd.read_csv(matches[0])
            
            # กรองให้เหลือเฉพาะวันที่ที่มีร่วมกันกับ GAN เพื่อความยุติธรรม
            df_filtered = df[df['Date'].isin(common_dates)].copy()
            
            if len(df_filtered) == 0:
                print(f"   ⚠️ No overlapping dates for {model_name}")
                continue

            # คำนวณ Metrics ทั้งหมดตามลำดับ: RMSE, MSE, MAE, MAPE, DA, Precision, Recall, F1
            actual_vals = df_filtered['Actual'].values
            predicted_vals = df_filtered['Predicted'].values
            
            metrics_results = calculate_all_metrics(actual_vals, predicted_vals)
            
            # เก็บข้อมูลลงตาราง
            row = {
                'Ticker': ticker,
                'Model': model_name,
                'Data_Points': len(df_filtered)
            }
            row.update(metrics_results)
            final_report_data.append(row)

    # 3. บันทึกผลลัพธ์
    if final_report_data:
        final_df = pd.DataFrame(final_report_data)
        
        # สร้างลำดับสำหรับการเรียง Model (Categorical Sort)
        final_df['Model'] = pd.Categorical(final_df['Model'], categories=model_folders, ordered=True)
        
        # เรียงลำดับตาม Ticker (A-Z) และตาม Model (1-6)
        final_df = final_df.sort_values(['Ticker', 'Model'])
        
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Evaluation Completed! Summary saved to: {OUTPUT_FILE}")
        print(f"📊 Order: 1. Tickers (Alphabetical), 2. Models (Model1-Model6)")
    else:
        print("❌ No data available for evaluation.")

if __name__ == "__main__":
    run_evaluation()
