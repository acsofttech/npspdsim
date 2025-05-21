# 🧠 NPS Robotics | PD Line-Follower Simulator Game

จำลองการทำงานของหุ่นยนต์เดินตามเส้นแบบสมจริง  
พร้อมระบบควบคุม PD Controller และเซ็นเซอร์ 8 ตัวในโลกเสมือนจริง 🎯  
พัฒนาโดย NPS Robotics เพื่อการเรียนรู้ระบบควบคุมหุ่นยนต์อย่างแท้จริง

---

## ✅ Features

- ปรับค่า **Base Speed / Kp / Kd** ได้แบบเรียลไทม์
- ปรับ **ขนาดการแสดงผลของสนาม** ได้
- ใช้ **Reflective Sensors จำนวน 8 ตัว**
- ปรับ **ระยะห่างของเซ็นเซอร์** ได้อิสระ
- มีระบบ **Checkpoint และ Finish Line**
- ตรวจจับ **การหลุดเส้น** อัตโนมัติ
- แสดงผล **เวลาวิ่งครบรอบ (Lap Timing)**
- รองรับการโหลดสนามจากไฟล์ภาพ PNG (ขนาดแนะนำ 800×800 px)
- จำลองการทำงานของหุ่นยนต์บนเบราว์เซอร์ โดยไม่ต้องต่อกับฮาร์ดแวร์จริง

---

## 🎯 สีมาตรฐานของสนาม

- **Checkpoint 1** ➜ `#FF00FF` (ม่วงชมพู)
- **Checkpoint 2** ➜ `#FF8000` (ส้ม)
- **Finish Line** ➜ `#0000FF` (น้ำเงิน)

---

## 🚀 วิธีติดตั้งและใช้งาน

### 1. Clone โครงการ
```bash
git clone https://github.com/acsofttech/npspdsim.git
cd npspdsim
```

### 2.สร้าง Virtual Environment
```bash
python -m venv venv
# สำหรับ macOS/Linux:
source venv/bin/activate
# สำหรับ Windows:
venv\Scripts\activate
```

3. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

4. รันแอปพลิเคชัน
```bash
streamlit run app.py
```

🧪 Technologies Used
Python 3.x
Streamlit
NumPy
Pillow (PIL)
