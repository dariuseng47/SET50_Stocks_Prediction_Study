import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf # สำหรับดึงข้อมูลหุ้น
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# 1. การตั้งค่าหน้าจอ
st.set_page_config(page_title="SET50 Stock Prediction", layout="wide")

st.title("📈 SET50 Stocks Prediction Study")
st.subheader("ระบบทำนายราคาหุ้นกลุ่ม SET50 ด้วย Deep Learning")

# 2. Sidebar สำหรับเลือกหุ้น
with st.sidebar:
    st.header("การตั้งค่า")
    stock_symbol = st.selectbox(
        "เลือกหุ้นที่ต้องการ (SET50):",
        ["PTT.BK", "AOT.BK", "CPALL.BK", "ADVANC.BK", "KBANK.BK"] # ตัวอย่าง
    )
    period = st.slider("จำนวนวันที่ใช้ย้อนหลัง (Days):", 30, 365, 90)

# 3. ส่วนการดึงข้อมูล
st.write(f"### ข้อมูลหุ้น: {stock_symbol}")
@st.cache_data # ใช้ Cache เพื่อให้โหลดเร็วขึ้น
def load_data(symbol):
    data = yf.download(symbol, period='1y')
    return data

data_load_state = st.text('กำลังดึงข้อมูลจาก Yahoo Finance...')
df = load_data(stock_symbol)
data_load_state.text('ดึงข้อมูลสำเร็จ!')

# 4. แสดงกราฟราคาปัจจุบัน
st.line_chart(df['Close'])

# 5. ส่วนการทำนาย (Prediction)
st.divider()
st.write("### 🤖 ผลการทำนายราคา")

try:
    # สมมติว่าคุณมีไฟล์ model อยู่ใน repo
    # model = load_model('my_model.h5') 
    st.info("ส่วนการประมวลผลโมเดล: กำลังเตรียมการแสดงผลการทำนาย...")
    
    # ตัวอย่างการแสดงผล (Place holder)
    col1, col2 = st.columns(2)
    col1.metric("ราคาปิดล่าสุด", f"{df['Close'].iloc[-1]:.2f}", "-0.50")
    col2.metric("ราคาทำนายพรุ่งนี้", "รอการคำนวณ...", "0.00")

except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")

st.caption("พัฒนาโดย: KhunPsc | พลังงานสะอาดเพื่ออนาคต")
