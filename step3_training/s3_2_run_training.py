import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import gc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error

# เพิ่ม Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# แก้ไขชื่อ Import ให้ถูกต้องตามชื่อไฟล์ที่เปลี่ยนใหม่
from step1_data_import.s1_4_preprocessor import preprocess_stock_data, get_walk_forward_data
from step3_training.s3_1_wgan_gp_trainer import WGANGPTrainerV1
from step5_evaluation.s5_1_metrics import calculate_all_metrics

# --- Configuration ---
TICKER_LIST_FILE = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step1_data_import\ticker_list.csv'
RAW_DATA_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step1_data_import\raw_data'
MODEL_ARCH_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step2_model_architecture\2_train_test_implementations'
MODEL_SAVE_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step2_model_architecture\saved_models'
RESULT_DIR = r'D:\Nueral Network\Thai_Stock_Prediction\3rd Study\step4_result'

def select_model():
    if len(sys.argv) > 1:
        model_files = sorted(glob.glob(os.path.join(MODEL_ARCH_DIR, "*.PY")))
        model_names = [os.path.basename(f).replace('.PY', '') for f in model_files]
        choice = sys.argv[1]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_names):
                selected_model_filename = model_names[idx]
                module_name = f"step2_model_architecture.2_train_test_implementations.{selected_model_filename}"
                selected_module = __import__(module_name, fromlist=['create_model', 'create_lstm_model', 'create_cnn_lstm_model'])
                return selected_model_filename, selected_module
        except ValueError: pass

    model_files = sorted(glob.glob(os.path.join(MODEL_ARCH_DIR, "*.PY")))
    model_names = [os.path.basename(f).replace('.PY', '') for f in model_files]
    print("\n--- 🏗️ Available Models (1-6) ---")
    for i, name in enumerate(model_names, 1):
        display_name = name.split('_', 1)[1] if '_' in name else name
        print(f"{i}. {display_name}")
    
    while True:
        try:
            choice = input("\nเลือกหมายเลขโมเดลที่ต้องการเทรน: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(model_names):
                selected_model_filename = model_names[idx]
                module_name = f"step2_model_architecture.2_train_test_implementations.{selected_model_filename}"
                selected_module = __import__(module_name, fromlist=['create_model', 'create_lstm_model', 'create_cnn_lstm_model'])
                return selected_model_filename, selected_module
            else: print("⚠️ หมายเลขไม่ถูกต้อง")
        except ValueError: print("⚠️ กรุณาใส่เป็นตัวเลข")

def select_features():
    if len(sys.argv) > 2:
        user_input = sys.argv[2]
        if user_input.lower() == 'default': return ['Close']
        return [s.strip() for s in user_input.split(',')]

    # กรองเฉพาะไฟล์ .csv เท่านั้น
    csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("❌ ไม่พบไฟล์ CSV ในโฟลเดอร์ข้อมูล")
        return ['Close']
        
    sample_file = csv_files[0]
    df_sample = pd.read_csv(os.path.join(RAW_DATA_DIR, sample_file))
    available_cols = [c for c in df_sample.columns if c not in ['Date', 'Unnamed: 0']]
    
    print("\n--- 📊 Available Input Features ---")
    for i, col in enumerate(available_cols, 1): print(f"{i}. {col}")
    
    user_input = input("\nระบุฟีเจอร์ [Default: Close]: ").strip()
    if not user_input: return ['Close']
    selected = [s.strip() for s in user_input.split(',')]
    return [s for s in selected if s in available_cols] or ['Close']

def run_main_training():
    model_filename, model_module = select_model()
    model_base_name = model_filename
    selected_features = select_features()
    num_features = len(selected_features)
    is_gan = "GAN" in model_base_name.upper()
    
    ticker_filter = sys.argv[3] if len(sys.argv) > 3 else None
    
    tickers_df = pd.read_csv(TICKER_LIST_FILE)
    tickers = tickers_df['Ticker'].tolist()

    if ticker_filter:
        ticker_filter_upper = ticker_filter.upper()
        # กรองแบบ Exact Match (รองรับทั้งกรณีใส่ .BK หรือไม่ใส่)
        tickers = [t for t in tickers if t.upper() == ticker_filter_upper or t.upper().split('.')[0] == ticker_filter_upper]
        if not tickers:
            print(f"❌ No tickers matching: {ticker_filter}")
            return

    lookback = 5
    noise_dim = 100
    batch_size = 32
    wf_window = 252 
    wf_step = 21    

    model_result_dir = os.path.join(RESULT_DIR, model_base_name)
    os.makedirs(model_result_dir, exist_ok=True)

    for ticker in tickers:
        print(f"\n--- 📈 Processing Ticker: {ticker} ---")
        clean_name = ticker.replace('.', '_')
        file_path = os.path.join(RAW_DATA_DIR, f"{clean_name}_stock_data.csv")
        
        # Check if result already exists to skip (Resume Logic)
        final_pred_file = os.path.join(model_result_dir, f"{clean_name}_predictions.csv")
        if os.path.exists(final_pred_file):
            print(f"⏭️ Skipping {ticker} - Results already exist.")
            continue

        # --- 🕒 Calculate Alignment Date (To match GAN's 20% Test Set) ---
        df_raw = pd.read_csv(file_path, parse_dates=['Date'])
        total_seqs = len(df_raw) - lookback
        train_len = int(total_seqs * 0.8) # GAN uses 0.8 static split
        gan_test_start_date = df_raw.iloc[lookback + train_len]['Date']
        # ---------------------------------------------------------------
        
        ticker_model_save_dir = os.path.join(MODEL_SAVE_DIR, model_base_name)
        os.makedirs(ticker_model_save_dir, exist_ok=True)
        save_path = os.path.join(ticker_model_save_dir, f"{clean_name}_best.keras")

        if not is_gan:
            all_step_preds, all_step_actuals, all_step_dates = [], [], []
            step_gen = get_walk_forward_data(file_path, wf_window, wf_step, lookback, selected_features)
            
            for step_idx, (X_train_full, y_train_full, X_test, y_test_actual, scaler_y, dates) in enumerate(step_gen):
                input_shape = (lookback, num_features)
                
                # Clear session to prevent memory leak
                tf.keras.backend.clear_session()
                gc.collect()

                if "LSTM" in model_base_name and "CNN" not in model_base_name:
                    model = model_module.create_lstm_model(input_shape)
                else:
                    model = model_module.create_cnn_lstm_model(input_shape)
                
                early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = model.fit(X_train_full, y_train_full, epochs=100, batch_size=batch_size, validation_split=0.125, callbacks=[early_stop], verbose=0)
                
                # บันทึกประวัติการเทรน (สำหรับ Step แรก หรือ Step สุดท้ายก็ได้ ในที่นี้บันทึกทุก Step ทับกันไป)
                history_df = pd.DataFrame(history.history)
                history_df.to_csv(os.path.join(model_result_dir, f"{clean_name}_training_history.csv"), index=False)

                preds_actual = scaler_y.inverse_transform(model.predict(X_test, verbose=0))
                all_step_preds.append(preds_actual.flatten())
                all_step_actuals.append(y_test_actual.flatten())
                all_step_dates.extend(dates)
                
                del model
                gc.collect()

            actuals_flat = np.concatenate(all_step_actuals)
            preds_flat = np.concatenate(all_step_preds)
            
            # --- 🕒 Filter Baseline Results to match GAN's 20% Evaluation Period ---
            full_results_df = pd.DataFrame({'Date': all_step_dates, 'Actual': actuals_flat, 'Predicted': preds_flat})
            results_df = full_results_df[full_results_df['Date'] >= gan_test_start_date].copy()
            
            # Use filtered results for CSV and Metrics
            results_df.to_csv(final_pred_file, index=False)
            
            # คำนวณ Metrics ทั้งหมดบนข้อมูลที่กรองแล้ว
            metrics = calculate_all_metrics(results_df['Actual'].values, results_df['Predicted'].values)
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            with open(os.path.join(model_result_dir, "evaluation_results.txt"), "a") as f:
                f.write(f"Ticker: {ticker}, {metrics_str}\n")
            
        else:
            # For GANs, memory leak is less severe but still good practice to clear
            tf.keras.backend.clear_session()
            gc.collect()

            X_train_full, y_train_full, X_test, y_test_actual, scaler_X, scaler_y, dates = preprocess_stock_data(
                file_path, lookback=lookback, feature_cols=selected_features
            )
            if X_train_full is None: continue

            _, generator, discriminator = model_module.create_model((noise_dim,), (lookback, num_features), (1, num_features), 1, num_features)
            trainer = WGANGPTrainerV1(generator, discriminator)
            dataset = tf.data.Dataset.from_tensor_slices((X_train_full, y_train_full)).shuffle(1024).batch(batch_size, drop_remainder=True)
            
            best_mae, history = trainer.train(dataset, epochs=1000, noise_dim=noise_dim, batch_size=batch_size, save_path=save_path)
            
            # Save history
            history_df = pd.DataFrame(history)
            history_df.to_csv(os.path.join(model_result_dir, f"{clean_name}_training_history.csv"), index=False)
            
            test_noise = np.random.normal(0, 1, (len(X_test), noise_dim))
            gen_preds_scaled = generator.predict([test_noise, X_test], verbose=0)[:, 0, 0]
            preds_actual = scaler_y.inverse_transform(gen_preds_scaled.reshape(-1, 1)).flatten()
            
            y_test_flat = y_test_actual.flatten()
            results_df = pd.DataFrame({'Date': dates, 'Actual': y_test_flat, 'Predicted': preds_actual})
            results_df.to_csv(final_pred_file, index=False)
            
            # คำนวณ Metrics ทั้งหมด
            metrics = calculate_all_metrics(y_test_flat, preds_actual)
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            with open(os.path.join(model_result_dir, "evaluation_results.txt"), "a") as f:
                f.write(f"Ticker: {ticker}, {metrics_str}\n")
            
            del generator, discriminator, trainer
            gc.collect()
        
        print(f"✅ Finished {ticker}")

if __name__ == "__main__":
    run_main_training()
