# 🧪 Methodology: Thai Stock Prediction (3rd Study)

ไฟล์นี้บันทึกขั้นตอนและวิธีการดำเนินงานอย่างละเอียดในแต่ละขั้นตอนของโปรเจค 3rd Study

---

## 🛠️ ขั้นตอนที่ 1: การจัดเตรียมและประมวลผลข้อมูล (Data Preparation & Preprocessing)

เพื่อให้ข้อมูลมีความต่อเนื่องและแม่นยำในการวิเคราะห์เชิงสถิติ ผู้ศึกษาได้กำหนดเกณฑ์ในการคัดเลือกหลักทรัพย์จากดัชนี SET50 ย้อนหลัง 5 ปี ระหว่างวันที่ 19 กรกฎาคม 2019 ถึง 19 กรกฎาคม 2024 โดยมีรายละเอียดการพิจารณา ดังนี้:

### 1.1 ความต่อเนื่องของหลักทรัพย์
คัดเลือกเฉพาะหลักทรัพย์ที่ไม่มีการออกและเสนอขายหุ้นใหม่แก่ประชาชนทั่วไปเป็นครั้งแรก (IPO), ไม่มีการแตกพาร์ หรือการเปลี่ยนสัญลักษณ์การซื้อขายในช่วงเวลาดังกล่าว ทั้งนี้ หากมีการเปลี่ยนสัญลักษณ์แต่ข้อมูลจาก Yahoo Finance ยังคงแสดงค่าต่อเนื่องและครบถ้วน ผู้ศึกษาจะนำหลักทรัพย์นั้นมาคำนวณด้วย (คัด DELTA, CRC, OR ออกเนื่องจากไม่ผ่านเกณฑ์ความต่อเนื่อง)

### 1.2 ความสมบูรณ์ของข้อมูล
หลักทรัพย์ต้องมีจำนวนวันซื้อขายรวมเท่ากันที่ **1,212 วัน** เพื่อป้องกันความคลาดเคลื่อนในการเปรียบเทียบข้อมูลรายวัน (คัด SCB ออกเนื่องจากประวัติข้อมูลไม่ครบถ้วนตามเกณฑ์ความสมบูรณ์)

### 1.3 การปรับมาตราส่วนข้อมูล (Data Scaling)
ผู้ศึกษาเลือกใช้เทคนิค **MinMaxScaler** ในการปรับช่วงราคาหุ้น (Close Price) ให้อยู่ในช่วง **[-1, 1]** เพื่อให้มีความสอดคล้องกับ Activation Function ชนิด **Hyperbolic Tangent (tanh)** ของเครื่องมือสร้างข้อมูล (Generator) ในสถาปัตยกรรม GAN ซึ่งจะช่วยเพิ่มเสถียรภาพและประสิทธิภาพในการฝึกสอนโมเดล

### 1.4 ระยะเวลาการดูย้อนหลัง (Lookback Period)
กำหนดระยะเวลาการดูข้อมูลราคาหุ้นย้อนหลัง (**Lookback Period**) เท่ากับ **5 วันทำการ** เพื่อใช้เป็นชุดข้อมูลนำเข้า (Input Sequence) ในการวิเคราะห์แนวโน้มและทำนายราคาหุ้นในวันทำการถัดไป (t+1)

### 1.5 สรุปรายการหลักทรัพย์ที่ถูกคัดเลือก (Final Ticker List - 18 Tickers)
จากการพิจารณาตามเกณฑ์ข้างต้น มีหลักทรัพย์ที่ผ่านเกณฑ์จำนวนทั้งสิ้น **18 ตัว** ได้แก่:
1. ADVANC 2. AOT 3. BBL 4. BDMS 5. CPALL 6. CPN 7. GPSC 8. HMPRO 9. IVL 10. KBANK 11. KTB 12. MINT 13. PTTEP 14. PTT 15. SCC 16. TRUE 17. TTB 18. WHA

### 1.6 การวิศวกรรมข้อมูลและอินดิเคเตอร์เสริม (Feature Engineering)
ผู้ศึกษาได้คำนวณและเพิ่มข้อมูลทางเทคนิค (Reserved Features) เข้าไปในชุดข้อมูลดิบ เพื่อรองรับการทดลองแบบ Conditional GAN (cGAN) และการใช้งานระบบ Interactive Feature Selection ในอนาคต ได้แก่:
- **Trend Indicators:** SMA 5, 10 และ EMA 5, 10
- **Momentum Indicator:** RSI 14
- **Risk Indicator:** 5-day Standard Deviation (Volatility)
- **Calculated Returns:** Daily Return, Return %, และ Log Return

---

## 🏗️ ขั้นตอนที่ 2: การออกแบบสถาปัตยกรรมและการฝึกสอนโมเดล (Model Architecture & Training)

### 2.1 ลำดับสถาปัตยกรรมโมเดล (Model Architectures Order)
เพื่อให้การเปรียบเทียบผลลัพธ์เป็นระบบ ผู้ศึกษาได้จัดเรียงลำดับการทดลองโมเดลทั้ง 6 รูปแบบดังนี้:
1. **LSTM:** Long Short-Term Memory (Base Regressor)
2. **CNN-LSTM:** Hybrid CNN-LSTM Architecture
3. **CWGAN-GP:** Conditional Wasserstein GAN with Gradient Penalty
4. **LSTM-CWGAN-GP:** LSTM-based Generator with WGAN-GP
5. **LSTM-CNN-CWGAN-GP:** LSTM Generator and CNN Discriminator
6. **CNN-LSTM-CWGAN-GP:** CNN-LSTM Hybrid Generator

### 2.2 กลยุทธ์การตรวจสอบความถูกต้อง (Validation Strategy)
- **Baseline Models (1-2):** ใช้เทคนิค **Walk-forward Validation** (Window 252, Step 21) เพื่อจำลองการเทรนใหม่รายเดือนเสมือนการเทรดจริง
- **GAN Architectures (3-6):** ใช้เทคนิค **Static Split (80/20)** เพื่อรักษาสมดุลระหว่างความแม่นยำและทรัพยากรการคำนวณ

### 2.3 สัดส่วนการแบ่งชุดข้อมูล (Data Splitting)
เพื่อให้การทดสอบมีความน่าเชื่อถือ ผู้ศึกษาได้กำหนดสัดส่วนการแบ่งข้อมูลในแต่ละรอบการฝึกสอน ดังนี้:
- **ชุดข้อมูลฝึกสอน (Training Set):** 70% ของข้อมูลทั้งหมด
- **ชุดข้อมูลตรวจสอบ (Validation Set):** 10% ของข้อมูลทั้งหมด (แยกออกมาเพื่อ Monitor val_loss ระหว่างเทรน)
- **ชุดข้อมูลทดสอบ (Testing Set):** 20% ของข้อมูลทั้งหมด

### 2.4 พารามิเตอร์และการรักษาสมดุลในสถาปัตยกรรม GAN (WGAN-GP & L1 Balancing)
ผู้ศึกษาได้กำหนดตัวคูณน้ำหนัก (Lambda - λ) สองรูปแบบเพื่อรักษาสมดุลในการฝึกสอนโมเดล GAN ดังนี้:
- **1. Gradient Penalty Weight (λ_gp = 10.0):** 
    - *หน้าที่:* น้ำหนักสำหรับควบคุมความเสถียรของ Gradient (Stability Weight) 
    - *เหตุผล:* อิงตามมาตรฐาน Gulrajani et al. (2017) เพื่อบังคับเงื่อนไข Lipschitz Constraint
- **2. L1 Reconstruction Weight (λ_L1 = 100.0):** 
    - *หน้าที่:* น้ำหนักสำหรับควบคุมความแม่นยำของราคาทำนาย (Accuracy Weight) 
    - *เหตุผล:* เพื่อรักษาสมดุลระหว่างกราฟที่ดูเหมือนจริงกับตัวเลขราคาที่ถูกต้อง
- **n_critic (Training Ratio): 5** 
    - *เหตุผล:* เพื่อให้ Discriminator ฉลาดเพียงพอก่อนไปสอน Generator ในแต่ละก้าวการเทรน (ตามมาตรฐาน Gulrajani et al., 2017)

### 2.5 พารามิเตอร์เฉพาะรายโมเดล (Model-Specific Parameters)
1. **LSTM:** 2 LSTM Layers (50 units), 1 Dense Layer (25 units), Dropout 0.2
2. **CNN-LSTM:** 1 Conv1D (64 filters, kernel 3), 2 LSTM (50 units), 1 Dense (25 units), Dropout 0.2
3. **CWGAN-GP:** Dense Generator (128, 256 units), Dense Discriminator (128, 64 units), Dropout 0.3
4. **LSTM-CWGAN-GP:** LSTM Generator (3 layers, 50 units), LSTM Discriminator (2 layers, 50 units), Dropout 0.2
5. **LSTM-CNN-CWGAN-GP:** LSTM Generator (3 layers, 50 units), CNN Discriminator (1 Conv1D, 64 filters), Dropout 0.3
6. **CNN-LSTM-CWGAN-GP:** Hybrid Generator (Dense Input 128 units + Conv1D), LSTM Discriminator, Dropout 0.3

---

## 📈 ขั้นตอนที่ 3: ผลลัพธ์และการประเมินประสิทธิภาพ (Results & Evaluation)

### 3.1 การจัดเก็บผลลัพธ์ (Results Management)
ในขั้นตอนที่ 4 (**4_result**), ระบบจะจัดเก็บไฟล์ผลการทำนายในรูปแบบ CSV แยกตามประเภทโมเดลและรายชื่อหลักทรัพย์ เพื่อใช้เป็นข้อมูลดิบสำหรับการวิเคราะห์เชิงลึก

### 3.2 การประเมินผลเชิงปริมาณ (Performance Metrics)
ในขั้นตอนที่ 5 (**5_evaluation**), ผู้ศึกษาได้กำหนดตัวชี้วัดประสิทธิภาพโดยเรียงลำดับดังนี้:
1. **RMSE**, 2. **MSE**, 3. **MAE**, 4. **MAPE (%)**, 5. **DA (%)**, 6. **Precision**, 7. **Recall**, 8. **F1-Score**

### 3.3 การแสดงผลเชิงทัศน์ (Visualization)
ในขั้นตอนที่ 6 (**6_visualization**), ข้อมูลจะถูกประมวลผลออกมาในรูปแบบภาพ 5 ประเภท:
1. **Prediction Plots:** กราฟเปรียบเทียบราคาจริงและราคาพยากรณ์ รายหุ้นรายโมเดล
2. **Stock Tables:** ตารางสรุป Metrics เปรียบเทียบทั้ง 6 โมเดล แยกเป็นรายหลักทรัพย์
3. **Global Summary (Leaderboard):** กราฟสรุปจำนวนครั้งที่แต่ละโมเดลทำค่าประสิทธิภาพได้ดีที่สุด
4. **Training Curves:** กราฟประวัติค่าความสูญเสีย (Loss History) เพื่อตรวจสอบความเสถียรในการฝึกสอน
5. **Error Analysis:** การวิเคราะห์ความคลาดเคลื่อนผ่าน **Scatter Plot** (Actual vs Predicted) และ **Error Distribution** (Histogram)
