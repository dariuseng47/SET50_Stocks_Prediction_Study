import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import OrderedDict

def calculate_all_metrics(actual, predicted):
    """
    คำนวณและส่งคืน Metrics เรียงลำดับตามที่กำหนด:
    RMSE, MSE, MAE, MAPE, DA, Precision, Recall, F1-Score
    """
    # 1. กลุ่ม Error Metrics
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100

    # 2. กลุ่ม Directional Metrics (คำนวณทิศทาง t เทียบกับ t-1)
    actual_diff = actual[1:] - actual[:-1]
    act_dir = (actual_diff >= 0).astype(int)
    
    pred_diff = predicted[1:] - actual[:-1]
    pred_dir = (pred_diff >= 0).astype(int)
    
    da = accuracy_score(act_dir, pred_dir) * 100
    precision = precision_score(act_dir, pred_dir, zero_division=0)
    recall = recall_score(act_dir, pred_dir, zero_division=0)
    f1 = f1_score(act_dir, pred_dir, zero_division=0)

    # จัดเรียงลำดับตามความต้องการของ User
    results = OrderedDict([
        ('RMSE', rmse),
        ('MSE', mse),
        ('MAE', mae),
        ('MAPE(%)', mape),
        ('DA(%)', da),
        ('Precision', precision),
        ('Recall', recall),
        ('F1-Score', f1)
    ])
    
    return results

def get_full_evaluation(file_path):
    """อ่านไฟล์ CSV และส่งคืนผลลัพธ์ที่คำนวณแล้ว"""
    try:
        df = pd.read_csv(file_path)
        actual = df['Actual'].values
        predicted = df['Predicted'].values
        return calculate_all_metrics(actual, predicted)
    except Exception as e:
        print(f"Error evaluating {file_path}: {e}")
        return None

if __name__ == "__main__":
    print("Evaluation Metrics Module v1.2 Ready.")
    print("Order: RMSE, MSE, MAE, MAPE, DA, Precision, Recall, F1-Score")
