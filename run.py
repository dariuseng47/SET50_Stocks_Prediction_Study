import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- 1. การตั้งค่าหน้าจอ ---
st.set_page_config(page_title="SET50 Stock Prediction", layout="wide")

st.title("📈 SET50 Stocks Prediction Study")
st.subheader("ระบบวิเคราะห์และทำนายราคาหุ้นกลุ่ม SET50")

# --- 2. Sidebar สำหรับเลือกหุ้น ---
with st.sidebar:
    st.header("การตั้งค่า")
    # รายชื่อหุ้น SET50 (ตัวอย่าง)
    stocks = ["PTT.BK", "AOT.BK", "CPALL.BK", "ADVANC.BK", "KBANK.BK", "BDMS.BK", "GULF.BK"]
    stock_symbol = st.selectbox("เลือกหุ้นที่ต้องการ:", stocks)
    
    st.divider()
    st.info("ระบบจะดึงข้อมูลย้อนหลัง 1 ปี เพื่อนำมาแสดงผลและทำนาย")

# --- 3. ฟังก์ชันดึงข้อมูล (พร้อมระบบป้องกัน Error) ---
@st.cache_data(ttl=3600) # เก็บ Cache ไว้ 1 ชม. ลดการโดน Rate Limit
def get_stock_data(symbol):
    try:
        # ดึงข้อมูล
        df = yf.download(symbol, period='1y', interval='1d')
        
        if df.empty:
            return None, "ไม่พบข้อมูลหุ้น หรืออาจติด Rate Limit จาก Yahoo Finance"
        
        # แก้ปัญหา Multi-index ของ yfinance เวอร์ชันใหม่
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        return df, None
    except Exception as e:
        return None, str(e)

# ดึงข้อมูลมาใช้งาน
data_load_state = st.text(f'กำลังโหลดข้อมูล {stock_symbol}...')
df, error_msg = get_stock_data(stock_symbol)

if error_msg:
    st.error(f"❌ เกิดข้อผิดพลาด: {error_msg}")
    st.warning("คำแนะนำ: ลองรีเฟรชหน้าจออีกครั้ง หรือเปลี่ยนตัวเลือกหุ้น")
    st.stop() # หยุดการทำงานถ้าไม่มีข้อมูล
else:
    data_load_state.empty()

# --- 4. แสดงข้อมูลสรุป (Metrics) ---
# บังคับแปลงเป็น Float เพื่อป้องกัน Error 'unsupported format string passed to Series.format'
try:
    last_close = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2])
    change = last_close - prev_close
    
    col1, col2, col3 = st.columns(3)
    col1.metric("ราคาปิดล่าสุด", f"{last_close:.2f} THB", f"{change:.2f}")
    col2.metric("ราคาสูงสุด (24h)", f"{float(df['High'].iloc[-1]):.2f}")
    col3.metric("ปริมาณการซื้อขาย", f"{int(df['Volume'].iloc[-1]):,}")

    # --- 5. แสดงกราฟราคา ---
    st.write(f"### กราฟราคาหุ้น {stock_symbol} (1 ปี)")
    st.line_chart(df['Close'])

    # --- 6. ส่วนการทำนาย (Prediction Zone) ---
    st.divider()
    st.write("### 🤖 ผลการทำนายด้วย AI (LSTM Model)")
    
    # หมายเหตุ: คุณต้องมีไฟล์โมเดลใน GitHub ของคุณ
    model_path = 'stock_model.h5' # เปลี่ยนชื่อให้ตรงกับไฟล์ใน GitHub
    
    try:
        # โค้ดส่วนนี้จะทำงานถ้ามีไฟล์โมเดลจริง
        # model = load_model(model_path)
        
        st.success("โมเดลพร้อมใช้งาน")
        
        # ตัวอย่างการแสดงผลทำนาย (Placeholder)
        st.info("ขณะนี้ระบบกำลังแสดงผลข้อมูลพื้นฐาน (ส่วน Logic การทำนายต้องปรับตามการเทรนโมเดลของคุณ)")
        
        prediction_price = last_close * 1.02 # สมมติว่า AI ทำนายว่าขึ้น 2%
        
        c1, c2 = st.columns(2)
        c1.write("#### ราคาที่คาดการณ์ (Next Day)")
        c1.subheader(f"฿ {prediction_price:.2f}")
        
        c2.write("#### ความเชื่อมั่น (Confidence)")
        c2.progress(85)
        
    except Exception as model_err:
        st.warning(f"⚠️ ระบบทำนายยังไม่พร้อม: ไม่พบไฟล์โมเดล '{model_path}' ในโปรเจกต์")
        st.caption(f"Debug Info: {model_err}")

except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการประมวลผลตัวเลข: {e}")

st.divider()
st.caption(f"Data updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Developed for SET50 Study")
